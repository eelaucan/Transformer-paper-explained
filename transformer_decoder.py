import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module): # heart of the transformer lets every token look at every other token in the sequence
    def __init__(self, embed_size, heads): # init initializes the attributes that define how the multi-head self-attention works
        super(SelfAttention, self).__init__() # This calls the parent class (nn.Module)’s constructor. PyTorch needs this call to properly register parameters, submodules, etc.
        self.embed_size = embed_size # Without self., the variables only exist inside the function (__init__). They disappear after the function ends.
        self.heads = heads  # with self It attaches the variable to the object itself
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size must be divisible by heads"

        # Linear layers to get Queries, Keys, and Values
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # Final fully connected layer
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, embed_size = x.shape

        # Split embedding into heads
        Q = self.query(x).view(N, seq_length, self.heads, self.head_dim) # we change x.shape into this. (2, 8, 10, 32)
        K = self.key(x).view(N, seq_length, self.heads, self.head_dim)
        V = self.value(x).view(N, seq_length, self.heads, self.head_dim)

        # Transpose for multi-head attention: (N, heads, seq_len, head_dim)
        Q = Q.transpose(1, 2) #we do this so that we can easily compute the attention per head
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.transpose(-1, -2)) / (self.embed_size ** 0.5) # scaled dot-product attention
        attention = torch.softmax(energy, dim=-1) # For each token’s Query vector, you want to measure similarity with every other token’s Key vector.
        # we need to turn K into a matrix vector to compute a dot product. so we take the transpose and swap the last two dimensions. 
        # which gives us the attention matrix
        # Multiply attention scores with V
        out = torch.matmul(attention, V) # weighted sum of the value vectors for each token

        # Merge heads back together
        out = out.transpose(1, 2).contiguous().view(N, seq_length, embed_size) # flatten heads and head_dim

        # Final linear layer
        out = self.fc_out(out) # until here you glued outputs side by side. but now you need to build the
        # interactions between heads. this means the model can't combine these different "views" effectively 

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads) # multi-head self-attention mechanism. context aware representation 
        # layer normalization normalizing outputs to stabilize and speed up training.
        self.norm1 = nn.LayerNorm(embed_size) # one for after attention
        self.norm2 = nn.LayerNorm(embed_size) # one for after feed-forward
        self.feed_forward = nn.Sequential(  # feed forward network
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(), # expanded the embedding size by f. e. then applyed ReLU activation
            nn.Linear(forward_expansion * embed_size, embed_size) # and finally reduced back to embed size.
        )
        self.dropout = nn.Dropout(dropout) # randomly zeros out elements during training for regulatization

    def forward(self, x):
        # Residual connection + LayerNorm after attention
        attention = self.attention(x) # every tokens learn from all others
        x = self.norm1(attention + x) # we add original input back which is residual connection to normalize for stability. 
        x = self.dropout(x)

        # Residual connection + LayerNorm after feed-forward
        forward = self.feed_forward(x) # each tokens embedding is refined individually
        out = self.norm2(forward + x) # we again add input back and normalize
        out = self.dropout(out) # randomly zero-out parts for regularization during training.

        return out
    
class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_size, 
        num_layers, 
        heads, 
        dropout, 
        forward_expansion, 
        max_length
    ):
        super(TransformerDecoder, self).__init__()
        self.embed_size = embed_size
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([ # This creates a list of TransformerBlock layers. and it stores them in a ModuleList.
            TransformerBlock(embed_size, heads, dropout, forward_expansion)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, vocab_size) # This is a fully connected (linear) layer that maps:
        # From: embed_size (like 512) → To: vocab_size (like 30,000) 
        # After the stack of TransformerBlocks, each token is represented as a vector of size embed_size. But we want to predict a word.
        # So we need to turn that vector into a probability distribution over all vocabulary words.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length)
        # in positions we are creating position IDs for every token in every sequence.
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers: # self.layers is a list of stacked TransformerBlocks.
            out = layer(out)  # For each TransformerBlock, you pass the current out through it:
        # vectors now contain more context about the whole sequence. 

        out = self.fc_out(out)
        return out
# Hyperparameters
vocab_size = 10000    # size of vocabulary (number of unique tokens)
embed_size = 256      # size of embedding vector
num_layers = 6        # number of Transformer blocks
heads = 8             # number of attention heads
dropout = 0.1         # dropout probability
forward_expansion = 4 # expand hidden size in feed-forward
max_length = 100      # max length of input sequence

# Model
model = TransformerDecoder(
    vocab_size,
    embed_size,
    num_layers,
    heads,
    dropout,
    forward_expansion,
    max_length
) # builds a stack of 6 transformerblocks. 

# Dummy input (batch=2, sequence length=10)
x = torch.randint(0, vocab_size, (2, 10)) # creates a random tensor of shape 2,10. 
# each value is a random integer from 0 to 9999
out = model(x) # this runs the forward method of the model.

print(out.shape)  # (2, 10, vocab_size)