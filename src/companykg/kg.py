"""
Copyright (C) eqtgroup.com Ltd 2023
https://github.com/EQTPartners/CompanyKG
License: MIT, https://github.com/EQTPartners/CompanyKG/LICENSE.md
"""

import logging
import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import spatial
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

from companykg.settings import (
    EDGES_FILENAME,
    ZENODO_DATASET_BASE_URI,
    EDGES_WEIGHTS_FILENAME,
    NODES_FEATURES_FILENAME_TEMPLATE,
    EVAL_TASK_FILENAME_TEMPLATE,
)
from companykg.utils import download_zenodo_file

logger = logging.getLogger(__name__)


class CompanyKG:
    """The CompanyKG class that provides utility functions
    to load data and carry out evaluations.
    """

    eval_task_types = ("sp", "sr", "cr", "ep")

    def __init__(
        self,
        nodes_feature_type: str = "msbert",
        load_edges_weights: bool = False,
        data_root_folder: str = "./data",
    ) -> None:
        """Initialize a CompanyKG object.

        Args:
            nodes_feature_type (str, optional): the desired note feature type.
                Viable values include "msbert", "pause", "simcse", "ada2". Defaults to "msbert".
            load_edges_weights (bool, optional): load edge weights or not. Defaults to False.
            data_root_folder (str, optional): root folder of downloaded data. Defaults to "./data".
                If the folder does not exist, the latest version of the dataset will be downloaded from
                Zenodo.
        """

        self.data_root_folder = data_root_folder

        # Load nodes feature: only load one type
        self.nodes_feature_type = nodes_feature_type

        # Create a local data directory - NOP if directory already exists
        os.makedirs(data_root_folder, exist_ok=True)

        # Load edges
        # First check if edges file exists - download if it doesn't
        self.edges_file = os.path.join(data_root_folder, EDGES_FILENAME)
        if not os.path.exists(self.edges_file):
            download_zenodo_file(
                os.path.join(ZENODO_DATASET_BASE_URI, EDGES_FILENAME),
                self.edges_file,
            )
        self.edges = torch.load(self.edges_file)
        logger.info(f"[DONE] Loaded {self.edges_file}")

        # Load edge weights [Optional]
        # First check if edge weights file exists - download if it doesn't
        self.load_edges_weights = load_edges_weights
        if load_edges_weights:
            self.edges_weight_file = os.path.join(
                data_root_folder, EDGES_WEIGHTS_FILENAME
            )
            if not os.path.exists(self.edges_weight_file):
                download_zenodo_file(
                    os.path.join(ZENODO_DATASET_BASE_URI, EDGES_WEIGHTS_FILENAME),
                    self.edges_weight_file,
                )
            self.edges_weight = torch.load(self.edges_weight_file).to_dense()
            logger.info(f"[DONE] Loaded {self.edges_weight_file}")

        # Load nodes feaures file
        # Check for nodes features file - download if it doesn't exist
        _nodes_feature_filename = NODES_FEATURES_FILENAME_TEMPLATE.replace(
            "<FEATURE_TYPE>",
            nodes_feature_type,
        )
        self.nodes_feature_file = os.path.join(
            data_root_folder, _nodes_feature_filename
        )
        if not os.path.exists(self.nodes_feature_file):
            download_zenodo_file(
                os.path.join(ZENODO_DATASET_BASE_URI, _nodes_feature_filename),
                self.nodes_feature_file,
            )
        self._load_node_features()
        logger.info(f"[DONE] Loaded {self.nodes_feature_file}")

        # Load evaluation test data
        self.eval_tasks = dict()
        for task_type in self.eval_task_types:
            # Check if evaluation test data exists - otherwise download it
            _eval_task_filename = EVAL_TASK_FILENAME_TEMPLATE.replace(
                "<TASK_TYPE>", task_type
            )
            _eval_task_file = os.path.join(data_root_folder, _eval_task_filename)
            if not os.path.exists(_eval_task_file):
                download_zenodo_file(
                    os.path.join(ZENODO_DATASET_BASE_URI, _eval_task_filename),
                    _eval_task_file,
                )
            self.eval_tasks[task_type] = (
                _eval_task_file,
                pd.read_parquet(_eval_task_file),
            )
            logger.info(f"[DONE] Loaded {_eval_task_file}")

        self.n_edges = self.edges.shape[0]
        if self.load_edges_weights:
            self.edges_weight_dim = self.edges_weight.shape[1]

        # Default Top-K for CR task
        self.eval_cr_top_ks = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    def _load_node_features(self):
        self.nodes_feature = torch.load(self.nodes_feature_file)
        if self.nodes_feature.dtype is not torch.float32:
            self.nodes_feature = self.nodes_feature.to(dtype=torch.float32)
        # Set Vars
        self.n_nodes = self.nodes_feature.shape[0]
        self.nodes_feature_dim = self.nodes_feature.shape[1]

    def change_feature_type(self, feature_type: str):
        if feature_type != self.nodes_feature_type:
            self.nodes_feature_type = feature_type
            self._load_node_features()

    @property
    def nodes_id(self) -> list:
        """Get an ordered list of node IDs.

        Returns:
            list: an ordered (ascending) list of node IDs.
        """
        return [i for i in range(self.n_nodes)]

    def describe(self) -> None:
        """Print key statistics of loaded data."""
        print(f"data_root_folder={self.data_root_folder}")
        print(f"n_nodes={self.n_nodes}, n_edges={self.n_edges}")
        print(f"nodes_feature_type={self.nodes_feature_type}")
        print(f"nodes_feature_dimension={self.nodes_feature_dim}")
        if self.load_edges_weights:
            print(f"edges_weight_dimension={self.edges_weight_dim}")
        for task_type in self.eval_task_types:
            print(f"{task_type}: {len(self.eval_tasks[task_type][1])} samples")

    def to_pyg(self):
        """
        Build a PyTorch-geometric graph from the loaded CompanyKG.

        """
        try:
            from torch_geometric.data import Data
        except ImportError as e:
            raise ImportError(
                "pytorch-geometric is not installed: please install to produce PyG graph"
            ) from e

        # Incxlude edges going in both directions, since PyG uses directed graphs
        edge_index = torch.concat([self.edges.T, self.edges[:, [1, 0]].T], dim=1)
        return Data(x=self.nodes_feature, edge_index=edge_index)

    def to_igraph(self):
        """
        Build an iGraph graph from the loaded CompanyKG.
        Requires iGraph to be installed.

        """
        try:
            import igraph as ig
        except ImportError as e:
            raise ImportError(
                "python-igraph is not installed: please install to produce iGraph graph"
            ) from e

        g = ig.Graph()
        g.add_vertices(self.n_nodes)
        # Names should be strings
        g.vs["name"] = [str(i) for i in self.nodes_id]

        logger.info("Building iGraph graph from edges")
        if self.load_edges_weights:
            # Convert tensors to Np arrays
            edge_weights = self.edges_weight.numpy()
            edges = self.edges.numpy()

            # Flatten the non-zero weights for each edge so we have a separate edge for each weight type
            nonzeros = np.nonzero(edge_weights)
            # These are just the column indices of the nonzeros
            types = nonzeros[1]
            # The weights for these separated edges are the flattened non-zero values
            weights = edge_weights[nonzeros]
            # The edges themselves are indexed by the row indices of the non-zero values
            # This repeats edges where there are multiple non-zero weight types
            edges = edges[nonzeros[0]]

            # Flatten the non-zero weights for each edge so we have a separate edge for each weight type
            attrs = {
                "type": types,
                "weight": weights,
            }
            g.add_edges(
                edges,
                attributes=attrs,
            )
        else:
            g.add_edges((i, j) for (i, j) in self.edges)
        return g
    
    def evaluate_ep(self, embed: torch.Tensor, n_trials: int = 3) -> list:
        """Evaluate the specified node embeddings on EP task.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated.
            n_trials: the number of trials to run the test.

        Returns:
            list: a list of dict containing overall AUC score on EP task 
            together with per-category AUC scores.
        """
        def prepare_data(ep_df, split, embed, node_id_names, et_col_names):
            split_df = ep_df[ep_df.split == split]
            _node_ids_tensor = torch.tensor(
                split_df[node_id_names].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values, 
                dtype=torch.long)
            labels = torch.tensor(
                split_df[et_col_names].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values, 
                dtype=torch.int32)
            features = torch.cat((
                embed[_node_ids_tensor[:, 0]], 
                embed[_node_ids_tensor[:, 1]]), dim=1)
            return features, labels

        ## Prep validation and test data for EP
        ep_df = self.eval_tasks["ep"][1]
        node_id_names = ['node_id0', 'node_id1']
        et_col_names = ['et2', 'et3', 'et4', 'et5', 'et8', 'et10', 'et14', 'et15']
        # Prepare validation data and label
        validation_feature, validation_label = prepare_data(ep_df, "validation", embed, node_id_names, et_col_names)
        # Prepare test data and label
        test_feature, test_label = prepare_data(ep_df, "test", embed, node_id_names, et_col_names)
        
        ## Run n_trials training using embed
        test_auc_scores = []
        for trial in range(n_trials):
            print(f'Trial {trial + 1}')
            ## Prep training data and label
            # filtering and binarizing
            filtered_edges_weight = (self.edges_weight > 0).int()[:, [1, 2, 3, 4, 7, 9, 13, 14]]
            rows_with_single_one = filtered_edges_weight.sum(dim=1) == 1
            filtered_edges_weight = filtered_edges_weight[rows_with_single_one]
            filtered_edges_weight_np = filtered_edges_weight.numpy()
            filtered_edges = self.edges[rows_with_single_one]
            # class balancing
            counts_of_ones = filtered_edges_weight_np.sum(axis=0)
            min_count = counts_of_ones.min()
            mask = np.zeros(filtered_edges_weight_np.shape[0], dtype=bool)
            for col in range(filtered_edges_weight_np.shape[1]):
                indices_with_ones = np.where(filtered_edges_weight_np[:, col] == 1)[0]
                if len(indices_with_ones) > min_count:
                    sampled_indices = np.random.choice(indices_with_ones, min_count, replace=False)
                else:
                    sampled_indices = indices_with_ones
                mask[sampled_indices] = True
            training_label = filtered_edges_weight[mask]
            balanced_filtered_edges = filtered_edges[mask]
            # feature concatenation
            training_feature = torch.cat((
                embed[balanced_filtered_edges[:, 0]], 
                embed[balanced_filtered_edges[:, 1]]), dim=1)
            
            ## Define a simple neural network: feel free to implement your model
            class MultiClassClassifier(nn.Module):
                def __init__(self, input_size, num_classes):
                    super(MultiClassClassifier, self).__init__()
                    self.fc1 = nn.Linear(input_size, 512)
                    self.fc2 = nn.Linear(512, 256)
                    self.fc3 = nn.Linear(256, num_classes)
                    self.relu = nn.ReLU()
                    self.dropout = nn.Dropout(0.5)

                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = self.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = self.fc3(x)
                    return x
                
            # Model evaluation function: feel free to implement your evaluation metrics
            def evaluate_model(model, dataloader, device):
                model.eval()
                all_labels = []
                all_outputs = []
                total_loss = 0.0
                with torch.no_grad():
                    for features, labels in dataloader:
                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features)
                        loss = nn.CrossEntropyLoss()(outputs, torch.argmax(labels, dim=1))
                        total_loss += loss.item() * features.size(0)
                        all_labels.append(labels.cpu().numpy())
                        all_outputs.append(outputs.cpu().numpy())
                all_labels = np.concatenate(all_labels, axis=0)
                all_outputs = np.concatenate(all_outputs, axis=0)
                avg_loss = total_loss / len(dataloader.dataset)
                return all_labels, all_outputs, avg_loss

            ## Train-Eval-Test flow: feel free to implement your training flow
            # hardware management
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            training_feature = training_feature.to(device)
            training_label = training_label.to(device)
            validation_feature = validation_feature.to(device)
            validation_label = validation_label.to(device)
            # define the dataset and dataloader
            batch_size = 64
            train_dataset = TensorDataset(training_feature, training_label)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            validation_dataset = TensorDataset(validation_feature, validation_label)
            validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
            # initialize the model and optimizer
            input_size = training_feature.shape[1]
            num_classes = training_label.shape[1]
            model = MultiClassClassifier(input_size, num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            # training loop with early stopping
            num_epochs = 100
            patience = 5
            best_val_auc = 0.0
            epochs_no_improve = 0
            early_stop = False
            for epoch in range(num_epochs):
                model.train()
                for features, labels in train_loader:
                    optimizer.zero_grad()
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    labels_single = torch.argmax(labels, dim=1)
                    loss = nn.CrossEntropyLoss()(outputs, labels_single)
                    loss.backward()
                    optimizer.step()
                # evaluate on validation data
                val_labels, val_outputs, val_loss = evaluate_model(model, validation_loader, device)
                val_labels_binarized = label_binarize(torch.argmax(torch.tensor(val_labels), dim=1), classes=range(num_classes))
                overall_val_auc = roc_auc_score(val_labels_binarized, val_outputs, average='macro', multi_class='ovr')
                per_class_val_auc = roc_auc_score(val_labels_binarized, val_outputs, average=None, multi_class='ovr')
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Overall Validation AUC-ROC: {overall_val_auc:.4f}")
                for i, auc in enumerate(per_class_val_auc):
                    print(f"Class {i} Validation AUC-ROC: {auc:.4f}")
                # early stopping check
                if overall_val_auc > best_val_auc:
                    best_val_auc = overall_val_auc
                    best_model_state = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("Early stopping")
                        early_stop = True
                        break
            if not early_stop:
                best_model_state = model.state_dict()
            # load the best model
            model.load_state_dict(best_model_state)
            # evaluate on test data
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            test_dataset = TensorDataset(test_feature, test_label)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_labels, test_outputs, _ = evaluate_model(model, test_loader, device)
            test_labels_binarized = label_binarize(torch.argmax(torch.tensor(test_labels), dim=1), classes=range(num_classes))
            overall_test_auc = roc_auc_score(test_labels_binarized, test_outputs, average='macro', multi_class='ovr')
            per_class_test_auc = roc_auc_score(test_labels_binarized, test_outputs, average=None, multi_class='ovr')
            print(f"Overall Test AUC-ROC: {overall_test_auc:.4f}")
            for i, auc in enumerate(per_class_test_auc):
                print(f"Class {i} Test AUC-ROC: {auc:.4f}")
            # append the results to the list
            test_auc_scores.append({
                'overall_test_auc': overall_test_auc,
                'per_class_test_auc': per_class_test_auc
            })

        ## Calc mean and std
        overall_test_auc = [d['overall_test_auc'] for d in test_auc_scores]
        per_class_test_auc = np.array([d['per_class_test_auc'] for d in test_auc_scores])
        overall_mean = np.mean(overall_test_auc)
        overall_std = np.std(overall_test_auc)
        per_class_mean = np.mean(per_class_test_auc, axis=0)
        per_class_std = np.std(per_class_test_auc, axis=0)

        return {
            "overall_mean": overall_mean,
            "overall_std": overall_std,
            "per_class_mean": per_class_mean,
            "per_class_std": per_class_std,
            "test_auc_scores": test_auc_scores
        }

    def evaluate_sp(self, embed: torch.Tensor) -> float:
        """Evaluate the specified node embeddings on SP task.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated.

        Returns:
            float: AUC score on SP task.
        """
        test_data = self.eval_tasks["sp"][1]
        gt = test_data["label"].tolist()
        pred = []
        for _, row in test_data.iterrows():
            node_embeds = (embed[row["node_id0"]], embed[row["node_id1"]])
            try:
                with np.errstate(invalid="ignore"):
                    pred.append(
                        1 - 0.5 * spatial.distance.cosine(node_embeds[0], node_embeds[1])
                    )
            except:
                print(row)
                raise
        return roc_auc_score(gt, pred)

    def evaluate_sr(self, embed: torch.Tensor, split: str = "validation") -> float:
        """Evaluate the specified node embeddings on SR task.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated.
            split (str): the split (validation/test) on which the evaluation will be run.

        Returns:
            float: Accuracy on SR task.
        """
        test_data = self.eval_tasks["sr"][1]
        test_data = test_data[test_data["split"] == split]
        gt = test_data["label"].tolist()
        pred = []
        for _, row in test_data.iterrows():
            query_embed = embed[row["target_node_id"]]
            candidate0_embed = embed[row["candidate0_node_id"]]
            candidate1_embed = embed[row["candidate1_node_id"]]
            with np.errstate(invalid="ignore"):
                _p1 = 1 - 0.5 * spatial.distance.cosine(query_embed, candidate0_embed)
                _p2 = 1 - 0.5 * spatial.distance.cosine(query_embed, candidate1_embed)
            pred.append(0) if _p1 >= _p2 else pred.append(1)
        return accuracy_score(gt, pred)

    @staticmethod
    def search_most_similar(
        target_embed: torch.Tensor, embed: torch.Tensor
    ) -> Tuple[np.array, np.array]:
        """Search top-K most similar nodes to a target node.

        Args:
            target_embed (torch.Tensor): the embedding of the target node.
            embed (torch.Tensor): the node embeddings to be searched from, i.e. candidate nodes.
            K (int, optional): the number of nodes to be returned as search result. Defaults to 50.

        Returns:
            Tuple[np.array, np.array]: the node IDs and the cosine similarity scores.
        """
        with np.errstate(invalid="ignore"):
            sims = np.dot(embed, target_embed) / (
                np.linalg.norm(embed, axis=1) * np.linalg.norm(target_embed)
            )
        # Reverse so the most similar is first
        max_ids = np.argsort(sims)[
            -2::-1
        ]  # remove target company (first element in the reversed array)
        return max_ids, sims[max_ids]

    def cr_top_ks(self, embed: torch.Tensor, ks: List[int]):
        """
        Evaluate CR (Competitor retrieval) as the average recall @ k for a number of
        different values of k.

        :param embed: the node embeddings to be evaluated
        :param ks: list of k-values to evaluate at
        :return:
        """
        test_data = self.eval_tasks["cr"][1]
        target_nids = list(test_data["target_node_id"].unique())
        target_nids.sort()
        # Collect recalls for different ks from each sample
        k_recalls = dict((k, []) for k in ks)

        for nid in target_nids:
            competitor_df = test_data[test_data["target_node_id"] == nid]
            competitor_nids = set(list(competitor_df["competitor_node_id"].unique()))
            # Get as many predictions as we'll need for the highest k
            res_nids, res_dists = CompanyKG.search_most_similar(embed[nid], embed)
            for k in sorted(ks):
                k_res_nids = set(res_nids[:k])
                common_set = k_res_nids & competitor_nids
                recall = len(common_set) / len(competitor_nids)
                k_recalls[k].append(recall)

        # Average the recalls over samples for each k
        recalls = [np.mean(k_recalls[k]) for k in sorted(ks)]
        return recalls

    def cr_top_k(self, embed: torch.Tensor, k: int = 50) -> float:
        """Evaluate CR (Competitor Retrieval) performance using top-K hit rate.
        This function will evaluate each target company in CR test set.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated
            k (int, optional): the number of nodes to be returned as search result. Defaults to 50.

        Returns:
            Tuple[float, list]: the overall hit rate and the per-target hit rate.
        """
        return self.cr_top_ks(embed, [k])[0]

    def evaluate_cr(self, embed: torch.Tensor) -> list:
        """Evaluate the specified node embeddings on CR task.

        Args:
            embed (torch.Tensor): the node embeddings to be evaluated.

        Returns:
            float: the list of tuples containing the CR results.
                The first element in each tuple is the overall hit rate for top-K.
        """
        return self.cr_top_ks(embed, self.eval_cr_top_ks)

    def evaluate(
        self,
        embeddings_file: str = None,
        embed: torch.Tensor = None,
        tasks: tuple = eval_task_types,
        silent: bool = False,
    ) -> dict:
        """Evaluate the specified embedding on all evaluation tasks: SP, SR and CR.
        When none parameters provided, it will evaluate the embodied nodes feature.

        Args:
            embeddings_file (str, optional): the path to the embedding file;
                it has highest priority. Defaults to None.
            embed (torch.Tensor, optional): the embedding to be evaluated;
                it has second highest priority. Defaults to None.
            tasks: the tasks to run.
            silent (bool): by default, evaluation results are printed to stdout;
                if True, nothing is output, you just get the results in the
                returned dict

        Returns:
            dict: a dictionary of evaluation results.
        """
        if embeddings_file is not None:
            try:
                embed = torch.load(embeddings_file)
            except:
                embed = torch.load(embeddings_file, map_location="cpu")
            result_dict = {"source": embeddings_file}
            if not silent:
                print(f"Evaluate Node Embeddings {embeddings_file}:")
        elif embed is not None:
            result_dict = {"source": f"embed {embed.shape}"}
            if not silent:
                print(f"Evaluate Custom Embeddings:")
        else:
            embed = self.nodes_feature
            result_dict = {"source": self.nodes_feature_type}
            if not silent:
                print(f"Evaluate Node Features {self.nodes_feature_type}:")
        # SP Task
        if "sp" in tasks:
            if not silent:
                print("Evaluate SP ...")
            result_dict["sp_auc"] = self.evaluate_sp(embed)
            if not silent:
                print("SP AUC:", result_dict["sp_auc"])
        else:
            print("SP evaluation skip!")
        # SR Task
        if "sr" in tasks:
            if not silent:
                print("Evaluate SR ...")
            result_dict["sr_validation_acc"] = self.evaluate_sr(embed)
            result_dict["sr_test_acc"] = self.evaluate_sr(embed, split="test")
            if not silent:
                print(
                    "SR Validation ACC:",
                    result_dict["sr_validation_acc"],
                    "SR Test ACC:",
                    result_dict["sr_test_acc"],
                )
        else:
            print("SR evaluation skip!")

        # CR Task
        if "cr" in tasks:
            if not silent:
                print(f"Evaluate CR with top-K hit rate (K={self.eval_cr_top_ks}) ...")
            result_dict["cr_topk_hit_rate"] = self.evaluate_cr(embed)
            if not silent:
                print("CR Hit Rates:", result_dict["cr_topk_hit_rate"])
        else:
            print("CR evaluation skip!")

        # EP Task
        if "ep" in tasks:
            if not silent:
                print(f"Evaluate EP ...")
            result_dict["ep_test_auc"] = self.evaluate_ep(embed)
        else:
            print("EP evaluation skip!")

        return result_dict

    def get_dgl_graph(self, work_folder: str) -> list:
        """Obtain a DGL graph. If it has not been built before, a new graph will be constructed,
        otherwise it will simply load from file in the specified working directory.

        Args:
            work_folder (str): the working directory of graph building.

        Returns:
            list: the built graph(s).
        """
        try:
            import dgl
        except ImportError as e:
            raise ImportError(
                "DGL is not installed. Please install to produce DGL graph"
            ) from e

        dgl_file = os.path.join(work_folder, f"dgl_{self.nodes_feature_type}.bin")
        if os.path.isfile(dgl_file):
            return dgl.data.utils.load_graphs(dgl_file)[0]
        else:
            graph_data = {
                ("_N", "_E", "_N"): self.edges.tolist(),
            }
            g = dgl.heterograph(graph_data)
            g.ndata["feat"] = self.nodes_feature
            if self.load_edges_weights:
                g.edata["weight"] = self.edges_weight
            dgl.data.utils.save_graphs(dgl_file, [g])
        return [g]
