# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import torch
import dgl
from scipy import sparse as sp
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import oddt


def get_data_funcs():
    return preprocess_fn, collate_fn


def preprocess_fn(key, ori_receptor, ligand, dataset_name):
    m1 = ligand
    m1.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(m1)
    # prepare ligand
    n1 = m1.GetNumAtoms()
    c1 = m1.GetConformers()[0]
    d1 = np.array(c1.GetPositions())
    adj1 = GetAdjacencyMatrix(m1) + np.eye(n1)
    H1 = get_atom_feature(m1, True)

    # prepare protein
    # remove atoms far away from the binding site
    cr = ori_receptor.GetConformers()[0]
    dr = np.array(cr.GetPositions())
    dist_lr = distance_matrix(d1, dr)
    min_dist_lr = dist_lr.min(axis=0)
    atoms_to_remove = [i for i, d in enumerate(min_dist_lr) if d > 8.0]

    edit_m2 = Chem.EditableMol(ori_receptor)
    for aid in reversed(atoms_to_remove):
        # start from higher atom IDs, because the higher ones will change if the lower ones are modified
        edit_m2.RemoveAtom(aid)
    m2 = edit_m2.GetMol()
    m2.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(m2)
    n2 = m2.GetNumAtoms()
    c2 = m2.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    adj2 = GetAdjacencyMatrix(m2) + np.eye(n2)
    H2 = get_atom_feature(m2, False)

    # aggregation
    edge_types = np.zeros((n1 + n2, n1 + n2), dtype=np.int64)
    edge_types[:n1, :n1] = 2  # in-ligand edge
    edge_types[n1:, n1:] = 3  # in-receptor edge
    edge_types[:n1, n1:] = 4  # receptor-ligand edge
    edge_types[n1:, :n1] = 4  # receptor-ligand edge
    row, col = np.diag_indices_from(edge_types)
    edge_types[row, col] = 1  # zero length edge

    # aggregation
    H = np.concatenate([H1, H2], 0)
    agg_adj1 = np.zeros((n1 + n2, n1 + n2))
    agg_adj1[:n1, :n1] = adj1
    agg_adj1[n1:, n1:] = adj2
    agg_adj2 = np.copy(agg_adj1)
    dm = distance_matrix(d1, d2)
    dm = np.where(dm <= 8.0, dm, 0.0)
    agg_adj2[:n1, n1:] = np.copy(dm)
    agg_adj2[n1:, :n1] = np.copy(np.transpose(dm))

    agg_adj3 = np.copy(agg_adj2)
    agg_adj3[:n1, :n1] = distance_matrix(d1, d1) * adj1  # multiply with adj1 to avoid a fully connected graph
    agg_adj3[n1:, n1:] = distance_matrix(d2, d2) * adj2

    # issameAA
    sameAA_adj = np.zeros((n1 + n2, n1 + n2))
    sameAA_adj[:n1, :n1] = 2  # in-ligand edge
    sameAA_adj[:n1, n1:] = 2  # receptor-ligand edge
    sameAA_adj[n1:, :n1] = 2  # receptor-ligand edge

    # detect_issameAA
    mol = oddt.toolkits.rdk.Molecule(m2)
    mol.protein = True
    id = mol.atom_dict["resid"]
    for i in range(n1, n1 + n2):
        for j in range(n1, n1 + n2):
            sameAA_adj[i][j] = (id[i - n1] == id[j - n1])

    # prepare dgl.DGLGraph
    whole_edge_list = torch.FloatTensor(agg_adj2).nonzero(as_tuple=True)
    receptor_edge_list_ = torch.FloatTensor(adj2).nonzero(as_tuple=True)
    receptor_edge_list = tuple(i + n1 for i in receptor_edge_list_)
    ligand_edge_list = torch.FloatTensor(adj1).nonzero(as_tuple=True)

    whole_edge_types = torch.LongTensor([one_of_k_encoding(m, [1, 2, 3, 4]) for m in edge_types[whole_edge_list]])
    receptor_edge_types = torch.LongTensor([one_of_k_encoding(m, [1, 2, 3, 4]) for m in edge_types[receptor_edge_list]])
    ligand_edge_types = torch.LongTensor([one_of_k_encoding(m, [1, 2, 3, 4]) for m in edge_types[ligand_edge_list]])

    whole_edge_sameAA = torch.LongTensor([one_of_k_encoding(m, [0, 1, 2]) for m in sameAA_adj[whole_edge_list]])
    receptor_edge_sameAA = torch.LongTensor([one_of_k_encoding(m, [0, 1, 2]) for m in sameAA_adj[receptor_edge_list]])
    ligand_edge_sameAA = torch.LongTensor([one_of_k_encoding(m, [0, 1, 2]) for m in sameAA_adj[ligand_edge_list]])

    whole_edge_feat3 = torch.FloatTensor(agg_adj3[whole_edge_list])
    receptor_edge_feat3 = torch.FloatTensor(agg_adj3[receptor_edge_list])
    ligand_edge_feat3 = torch.FloatTensor(agg_adj3[ligand_edge_list])

    # bond feature
    ligand_bond_features = get_bond_feature(m1)
    receptor_bond_features = get_bond_feature(m2)
    whole_bond_feature = []
    receptor_bond_feature = []
    ligand_bond_feature = []
    for i, j in list(zip(whole_edge_list[0].numpy(), whole_edge_list[1].numpy())):
        if i < n1 and j < n1:
            whole_bond_feature.append(ligand_bond_features[n1 * i + j])
            ligand_bond_feature.append(ligand_bond_features[n1 * i + j])
        elif i < n1 <= j:
            whole_bond_feature.append([0, 0, 0, 0, 0, 1] + [0, 0, 1] + [0, 0, 1] + [0, 0, 0, 0, 1])
        elif i >= n1 > j:
            whole_bond_feature.append([0, 0, 0, 0, 0, 1] + [0, 0, 1] + [0, 0, 1] + [0, 0, 0, 0, 1])
        else:
            whole_bond_feature.append(receptor_bond_features[n2 * (i - n1) + (j - n1)])
            receptor_bond_feature.append(receptor_bond_features[n2 * (i - n1) + (j - n1)])

    whole_edge_bond_feature = torch.LongTensor(np.array(whole_bond_feature))
    receptor_edge_bond_feature = torch.LongTensor(np.array(receptor_bond_feature))
    ligand_edge_bond_feature = torch.LongTensor(np.array(ligand_bond_feature))

    whole_graph = dgl.graph(whole_edge_list, num_nodes=n1 + n2)
    whole_node_features = torch.FloatTensor(H)
    whole_graph.ndata['feat'] = whole_node_features
    whole_graph.edata['type'] = whole_edge_types
    whole_graph.edata['sameAA'] = whole_edge_sameAA
    whole_graph.edata['distance'] = whole_edge_feat3.unsqueeze(-1)
    whole_graph.edata['bond_feat'] = whole_edge_bond_feature
    whole_graph.ndata['ligand_mask'] = torch.FloatTensor([1.] * n1 + [0.] * n2).unsqueeze(-1)

    ligand_graph = dgl.graph(ligand_edge_list, num_nodes=n1)
    ligand_node_features = torch.FloatTensor(H1)
    ligand_graph.ndata['feat'] = ligand_node_features
    ligand_graph.edata['type'] = ligand_edge_types
    ligand_graph.edata['sameAA'] = ligand_edge_sameAA
    ligand_graph.edata['distance'] = ligand_edge_feat3.unsqueeze(-1)
    ligand_graph.edata['bond_feat'] = ligand_edge_bond_feature

    receptor_graph = dgl.graph(receptor_edge_list_, num_nodes=n2)
    receptor_node_features = torch.FloatTensor(H2)
    receptor_graph.ndata['feat'] = receptor_node_features
    receptor_graph.edata['type'] = receptor_edge_types
    receptor_graph.edata['sameAA'] = receptor_edge_sameAA
    receptor_graph.edata['distance'] = receptor_edge_feat3.unsqueeze(-1)
    receptor_graph.edata['bond_feat'] = receptor_edge_bond_feature

    pos_enc_dim = 8
    whole_graph = laplacian_positional_encoding(whole_graph, pos_enc_dim)
    ligand_graph = laplacian_positional_encoding(ligand_graph, pos_enc_dim)
    receptor_graph = laplacian_positional_encoding(receptor_graph, pos_enc_dim)

    if dataset_name == 'dud-e' or dataset_name == 'muv':
        label = 1 if 'active' in key else 0
    elif dataset_name == 'lit-pcba':
        label = 1 if not 'inactive' in key else 0

    sample = {
        'whole_graph': whole_graph,
        'receptor_graph': receptor_graph,
        'ligand_graph': ligand_graph,
        'label': label,
    }
    return sample


def collate_fn(sample):
    return {
        'whole_graph': dgl.batch([s['whole_graph'] for s in sample]),
        'receptor_graph': dgl.batch([s['receptor_graph'] for s in sample]),
        'ligand_graph': dgl.batch([s['ligand_graph'] for s in sample]),
        'label': torch.LongTensor([s['label'] for s in sample]),
    }


# the following are from GT
def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt='csr').astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    if EigVec.shape[1] < pos_enc_dim + 1:
        PadVec = np.zeros((EigVec.shape[0], pos_enc_dim + 1 - EigVec.shape[1]), dtype=EigVec.dtype)
        EigVec = np.concatenate((EigVec, PadVec), 1)
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()
    return g


# the following are from GAT
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    # print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def ligand_atom_feature(m, atom_i):
    atom = m.GetAtomWithIdx(atom_i)
    feature = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'other']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()] + \
              one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    try:
        feature = feature + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                  [atom.HasProp('_ChiralityPossible')]
    except:
        feature = feature + [0, 0] + [atom.HasProp('_ChiralityPossible')]
    return np.array(feature)  # (11, 6, 2, 6, 1, 5, 2, 1) --> total 34


def receptor_atom_feature(m, m_resname_dict, atom_i):
    amino_acid_list = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
                       "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
    atom = m.GetAtomWithIdx(atom_i)
    feature = one_of_k_encoding_unk(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H', 'other']) + \
              one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()] + \
              one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    try:
        feature = feature + one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S']) + \
                  [atom.HasProp('_ChiralityPossible')]
    except:
        feature = feature + [0, 0] + [atom.HasProp('_ChiralityPossible')]
    try:
        feature = feature + one_of_k_encoding(m_resname_dict[atom_i], amino_acid_list)
    except:
        feature = feature + [0] * 20
    return np.array(feature)  # (11, 6, 2, 6, 1, 5, 2, 1, 20) --> total 54


def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    mol = oddt.toolkits.rdk.Molecule(m)
    mol.protein = True
    atom_resname_dict = mol.atom_dict["resname"]
    for i in range(n):
        if is_ligand:
            H.append(ligand_atom_feature(m, i))
        else:
            H.append(receptor_atom_feature(m, atom_resname_dict, i))
    H = np.array(H)
    if is_ligand:
        H = np.concatenate([H, np.zeros((n, 54))], 1)
    else:
        H = np.concatenate([np.zeros((n, 34)), H], 1)
    return H


def get_bond_feature(mol):
    feat = []
    n = mol.GetNumAtoms()
    N_index, C_index, resid = detect_peptide_bond(mol)
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    # 6 dim bond type
                    if ((resid[i] - resid[j] == 1 and i in N_index and j in C_index) or
                        (resid[i] - resid[j] == -1 and i in C_index and j in N_index)
                    ):
                        bond_feats = [0, 0, 0, 0, 1, 0]  # peptide bond
                    else:
                        bt = bond.GetBondType()
                        bond_feats = [
                            bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                            bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, 0, 0]
                    # 3 + 3 dim
                    bond_feats = bond_feats + one_of_k_encoding_unk(bond.GetIsConjugated(), [0, 1, "nonbond"]) + \
                                 one_of_k_encoding_unk(bond.IsInRing(), [0, 1, "nonbond"])
                    # 5 dim Stereo type
                    bond_feats = bond_feats + one_of_k_encoding_unk(
                        str(bond.GetStereo()),
                        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "nonbond"])
                else:
                    bond_feats = [0, 0, 0, 0, 0, 1] + [0, 0, 1] + [0, 0, 1] + [0, 0, 0, 0, 1]
            else:
                bond_feats = [0, 0, 0, 0, 0, 1] + [0, 0, 1] + [0, 0, 1] + [0, 0, 0, 0, 1]
            feat.append(bond_feats)
    return np.array(feat)  # 17 dim edge feature


def detect_peptide_bond(mol):
    mol = oddt.toolkits.rdk.Molecule(mol)
    mol.protein = True
    mol_N_list = mol.res_dict["N"]
    mol_C_list = mol.res_dict["C"]
    resid = mol.atom_dict['resid']
    try:
        assert len(mol_N_list) == len(mol_C_list)
    except:
        if len(mol_N_list) < len(mol_C_list):
            mol_C_list = mol_C_list[:len(mol_N_list)]
        else:
            mol_N_list = mol_N_list[:len(mol_C_list)]
    mol_symbol_list = mol.atom_dict["coords"]

    mol_N_index = []
    mol_C_index = []

    for i in range(len(mol_N_list)):
        mol_N_index.append(np.where(mol_symbol_list == mol_N_list[i])[0][0])
        mol_C_index.append(np.where(mol_symbol_list == mol_C_list[i])[0][0])

    return mol_N_index, mol_C_index, resid
