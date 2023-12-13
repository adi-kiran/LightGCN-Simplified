import torch
from LightGCNConvolution import LightGCNConv

class CF_LGCN_E(torch.nn.Module):
    def __init__(self, num_items, embedding_dim=64, num_layers=3, fusion_type="mean", alpha=None, **kwargs):
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        if alpha is None:
            alpha = 1. / (num_layers + 1)
        self.alpha = [[alpha] for i in range(num_layers+1)]
        self.item_embedding = torch.nn.Embedding(num_embeddings = num_items, embedding_dim = self.embedding_dim)
        self.convs = torch.nn.ModuleList([LightGCNConv() for _ in range(num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.item_embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()

    def get_embedding(self, edge_index, rev_edge_index):
        user_embeddings = []
        item_embeddings = []
        num_users, num_items = edge_index.sparse_sizes()
        assert self.num_items == num_items
        i_x = self.item_embedding.weight # Layer 0 = Item embedding
        u_x = torch.zeros([num_users,self.embedding_dim]) # infer num_users from edge index to make it inductive
        torch.nn.init.xavier_uniform_(u_x)
        out = u_x * self.alpha[0] if self.fusion_type=="mean" else u_x
        user_embeddings.append(out)
        for i in range(1,self.num_layers+1):
            if i%2==0:
                # Even numbered Layer = UE-GCN Layer, i.e., User Embedding used to calculate Item Embedding
                # x now holds user embedding, so we pass R^T (rev_edge_index) and it will calculate the item embedding
                i_x = self.convs[i-1](x=(u_x,i_x), edge_index = rev_edge_index, size=(num_items, num_users))
                out = i_x * self.alpha[i] if self.fusion_type=="mean" else i_x
                item_embeddings.append(out)
            else:
                # Odd numbered Layer = EU-GCN Layer, i.e., Item Embedding used to calculate User Embedding
                # x now holds item embedding so we pass R (edge_index) and it will calculate the user embedding
                u_x = self.convs[i-1](x=(i_x,u_x), edge_index=edge_index, size=(num_users, num_items))
                out = u_x * self.alpha[i] if self.fusion_type=="mean" else u_x
                user_embeddings.append(out)
        if self.fusion_type=="mean":
            stacked_user_embedding = torch.stack(user_embeddings,dim=1)
            final_user_embedding = torch.sum(stacked_user_embedding,dim=1)
            stacked_item_embedding = torch.stack(item_embeddings,dim=1)
            final_item_embedding = torch.sum(stacked_item_embedding,dim=1)
        else:
            final_user_embedding = torch.cat(user_embeddings)
            final_item_embedding = torch.cat(item_embeddings)
        return (final_user_embedding, final_item_embedding)


    def forward(self, edge_index, rev_edge_index):
        user_final_embedding, item_final_embedding = self.get_embedding(edge_index, rev_edge_index)
        return { "user_final_embedding": user_final_embedding, 
                 "item_final_embedding": item_final_embedding,
                 "item_initial_embedding": self.item_embedding.weight }
