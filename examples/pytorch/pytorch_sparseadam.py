import argparse
from collections import OrderedDict

import torch
from torch import nn, arange, cat, optim
import torch.nn.functional as F
import horovod.torch as hvd

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Test with Dense and Sparse Adam Optimizers')
parser.add_argument('--batch-size', type=int, default=2560, metavar='N',
                    help='input batch size for training (default: 2560)')
parser.add_argument('--input-size', type=int, default=10, metavar='N',
                    help='input test size for testing (default: 10)')
parser.add_argument('--force-cpu', action='store_true', default=False,
                    help='force cpu to do training')
args = parser.parse_args()

class EnumType:
    def __init__(self, name, maxCategories, embed_dim=0):
        self.name = name
        self.maxCategories = maxCategories
        self.embed_dim = embed_dim


class EmbeddingsCollection(nn.Module):
    def __init__(self, annotation, concat=True, use_batch_norm=True):
        super().__init__()

        self.output_size = 0
        self.embeddings = {}
        for k, embed in annotation.items():
            self.embeddings[k] = torch.nn.Embedding(
                embed.maxCategories,
                embed.embed_dim,
                sparse=True,
            )
            self.output_size += embed.embed_dim
        self.embeddings = torch.nn.ModuleDict(self.embeddings)
        self.concat = concat

        if use_batch_norm and concat:
            self.input_normalizer = torch.nn.BatchNorm1d(self.output_size)
        else:
            self.input_normalizer = None

    def forward(self, input: OrderedDict):
        embeds = []
        k: str
        v: nn.modules.sparse.Embedding
        for k, v in self.embeddings.items():
            embeds.append(v(input[k]))
        if self.concat:
            embeds = torch.cat(embeds, -1)
        if self.input_normalizer:
            return self.input_normalizer(embeds)
        return embeds


class OneHotEncoding(nn.Module):
    def __init__(self, maxSize):
        super().__init__()
        assert isinstance(maxSize, int)
        self.maxSize = maxSize

    def forward(self, x):
        result = torch.zeros(*x.shape, self.maxSize, device=x.device, dtype=torch.float32)
        result.scatter_(-1, x.unsqueeze(-1), 1.0)
        return result


class OneHotEncodingCollection(nn.Module):
    def __init__(self, annotation, use_batch_norm=True):
        super().__init__()

        self.output_size = 0
        self.one_hot = {}
        for k, embed in annotation.items():
            self.one_hot[k] = OneHotEncoding(embed.maxCategories)
            self.output_size += embed.maxCategories
        
        self.one_hot = torch.nn.ModuleDict(self.one_hot)
    
    def forward(self, input: OrderedDict):
        embeds = []
        k: str
        v: OneHotEncoding
        for k, v in self.one_hot.items():
            embeds.append(v(input[k]))
        return embeds


class MyInput(nn.Module):
    def __init__(self, annotation, use_bn=False):
        super().__init__()

        self.output_size = 0
        if 'embeddings' in annotation:
            self.embeddings = EmbeddingsCollection(annotation["embeddings"])
            self.output_size += self.embeddings.output_size

        if 'one_hot' in annotation:
            self.one_hot = OneHotEncodingCollection(annotation["one_hot"])
            self.output_size += self.one_hot.output_size

        if self.output_size == 0:
            raise ValueError("MyInput was not able to process " + str(annotation.keys()))

    def forward(self, buffer: OrderedDict):
        features: List[torch.Tensor] = []
        
        if hasattr(self, "embeddings"):
            features.append(self.embeddings(buffer["embeddings"]))
        
        if hasattr(self, "one_hot"):
            features += self.one_hot(buffer['one_hot'])

        return torch.cat(features, -1)


class MyLinear(nn.Linear):
    def forward(self, x):
        if len(self.weight.shape) == 2:
            return super().forward(x)
        else:
            return torch.baddbmm(self.bias.unsqueeze(-2), x, self.weight.transpose(-1, -2))


class MySequential(nn.Module):
    def __init__(self, input_size, layers=None,
                 use_dropout=False, dropout_rate=0.5, use_bn=False,
                 activation=nn.ReLU, activate_final=False):
        super().__init__()
        self.layers = []

        if isinstance(input_size, (tuple, list)):
            input_size, = input_size

        if not isinstance(activation, (tuple, list)):
            activation = [activation] * len(layers)
            if not activate_final:
                activation[-1] = None

        hidden_dim = input_size
        for l_dim, act in zip(layers or [], activation):
            self.layers.append(MyLinear(hidden_dim, l_dim))
            if act is not None:
                if use_bn:
                    self.layers.append(nn.BatchNorm1d(l_dim))
                self.layers.append(act())
                if use_dropout:
                    self.layers.append(nn.Dropout(dropout_rate))
            hidden_dim = l_dim
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        return self.layers(inputs)


class MyModel(nn.Module):
    def __init__(self, annotation: OrderedDict, layers=[512, 1], layer_size=512, num_heads=4,
                 t_size=128, use_dropout=False, use_bn=True, num_segments=48):

        super().__init__()
        annotation = annotation
        if not layers:
            layers = [layer_size, layer_size, 1]
        self.num_segments = num_segments
        self.input = MyInput(annotation)

        self.sequential = MySequential(self.input.output_size, layers=layers, use_dropout=use_dropout, use_bn=use_bn)
        self.sequential2 = MySequential(layers[-1]+self.num_segments+2, layers=[1], use_dropout=use_dropout, use_bn=use_bn,) 

    def forward(self, buffer):
        hot1 = buffer["one_hot"]["hot1"]
        hot1_ref = arange(self.num_segments).reshape(1, self.num_segments).float()
        hot0 = buffer["one_hot"]["hot0"]
        hot0_ref = arange(2).reshape(1, 2).float()
        if hot1.is_cuda:
            hot1_ref = hot1_ref.cuda()
            hot0_ref = hot0_ref.cuda()
        hot1_onehot = (hot1.float().unsqueeze(-1) == hot1_ref).float()
        hot0_onehot = (hot0.float().unsqueeze(-1) == hot0_ref).float()
        network = self.sequential(self.input(buffer))
        link = self.sequential2(cat([network, hot1_onehot, hot0_onehot], dim=1))
        return link.squeeze(-1)


def huber_loss(a, b, delta=20):
    err = (a - b).abs()
    mask = err < delta
    return (0.5 * mask * (err ** 2)) + ~mask * (err * delta - 0.5 * (delta ** 2))


# Horovod: initialize library.
hvd.init()
torch.manual_seed(hvd.rank())
device="cpu"

if torch.cuda.is_available():
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(hvd.rank())
    device=f"cuda:{hvd.local_rank()}" 
    if args.force_cpu:
        device="cpu"

print("Rank: ", hvd.rank(), ", Device: ", device)


input_size = args.input_size
batch_size = args.batch_size

if hvd.rank() == 0:
    print("input_size: ", input_size, ", batch_size", batch_size)

buffer = [OrderedDict() for _ in range(input_size)]
annotation = OrderedDict()

annotation["embeddings"] = OrderedDict()
for i in range(input_size):
    buffer[i]["embeddings"] = OrderedDict()

annotation["embeddings"]["name0"] = EnumType("name0", 2385, 12)
for i in range(input_size):
    buffer[i]["embeddings"]["name0"] = torch.randint(2385, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name1"] = EnumType("name1", 201, 8)
for i in range(input_size):
    buffer[i]["embeddings"]["name1"] = torch.randint(201, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name2"] = EnumType("name2", 201, 8)
for i in range(input_size):
    buffer[i]["embeddings"]["name2"] = torch.randint(201, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name3"] = EnumType("name3", 6, 3)
for i in range(input_size):
    buffer[i]["embeddings"]["name3"] = torch.randint(6, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name4"] = EnumType("name4", 19, 5)
for i in range(input_size):
    buffer[i]["embeddings"]["name4"] = torch.randint(19, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name5"] = EnumType("name5", 1441, 11)
for i in range(input_size):
    buffer[i]["embeddings"]["name5"] = torch.randint(1441, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name6"] = EnumType("name6", 201, 8)
for i in range(input_size):
    buffer[i]["embeddings"]["name6"] = torch.randint(201, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name7"] = EnumType("name7", 22, 5)
for i in range(input_size):
    buffer[i]["embeddings"]["name7"] = torch.randint(22, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name8"] = EnumType("name8", 156, 8)
for i in range(input_size):
    buffer[i]["embeddings"]["name8"] = torch.randint(156, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name9"] = EnumType("name9", 1216, 11)
for i in range(input_size):
    buffer[i]["embeddings"]["name9"] = torch.randint(1216, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name10"] = EnumType("name10", 9216, 14)
for i in range(input_size):
    buffer[i]["embeddings"]["name10"] = torch.randint(9216, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name11"] = EnumType("name11", 88999, 17)
for i in range(input_size):
    buffer[i]["embeddings"]["name11"] = torch.randint(88999, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name12"] = EnumType("name12", 941792, 20)
for i in range(input_size):
    buffer[i]["embeddings"]["name12"] = torch.randint(941792, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name13"] = EnumType("name13", 9405, 14)
for i in range(input_size):
    buffer[i]["embeddings"]["name13"] = torch.randint(9405, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name14"] = EnumType("name14", 83332, 17)
for i in range(input_size):
    buffer[i]["embeddings"]["name14"] = torch.randint(83332, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name15"] = EnumType("name15", 828767, 20)
for i in range(input_size):
    buffer[i]["embeddings"]["name15"] = torch.randint(828767, (batch_size,), dtype=torch.long) 

annotation["embeddings"]["name16"] = EnumType("name16", 945195, 20)
for i in range(input_size):
    buffer[i]["embeddings"]["name16"] = torch.randint(945195, (batch_size,), dtype=torch.long) 


annotation["one_hot"] = OrderedDict()
for i in range(input_size):
    buffer[i]["one_hot"] = OrderedDict()

annotation["one_hot"]["hot0"] = EnumType("hot0", 3) # one_hot doesn't use dimension
for i in range(input_size):
    buffer[i]["one_hot"]["hot0"] = torch.randint(3, (batch_size,), dtype=torch.long) 

annotation["one_hot"]["hot1"] = EnumType("hot1", 50) # one_hot doesn't use dimension
for i in range(input_size):
    buffer[i]["one_hot"]["hot1"] = torch.randint(50, (batch_size,), dtype=torch.long) 


for i in range(input_size):
    buffer[i]["labels"] = torch.rand(batch_size, dtype=torch.float64)


model = MyModel(annotation)

loss_function = huber_loss
if torch.cuda.is_available():
    model = model.to(device)
    #model.cuda()

sparse_params = []
dense_params = []
for k,v in model.named_parameters():
    if "input.embeddings.embeddings" in k: 
        sparse_params.append((k,v))
    else:
        dense_params.append((k,v))

optimizers = []
if len(dense_params) > 0:
    opt = optim.Adam([v for _,v in dense_params], lr=0.001)
    opt = hvd.DistributedOptimizer(opt, dense_params)
    optimizers.append(opt)
if len(sparse_params) > 0:
    opt = optim.SparseAdam([v for _,v in sparse_params], lr=0.001)
    opt = hvd.DistributedOptimizer(opt, sparse_params)
    optimizers.append(opt)

if hvd.rank() == 0:
    print(optimizers)


# Horovod: broadcast parameters & optimizer state.
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
for opt in optimizers:
    hvd.broadcast_optimizer_state(opt, root_rank=0)


model.train()

for epoch in range(10):
    
    for i in range(input_size):
        if hvd.rank() == 0:
            print("epoch: ", epoch, "batch: ", i)
        batch = buffer[i]
        for type in batch:
            if type != "labels":
                for name,tensor in batch[type].items():
                    # Cast current batch tensor to GPU
                    batch[type][name] = tensor.to(device)       
            else:
                batch["labels"] = batch["labels"].to(device)            
        
        for opt in optimizers:
            opt.zero_grad()

        batch_pred = model(batch)

        loss = loss_function(batch_pred, batch["labels"], delta=60)
        loss.mean().backward()
        for opt in optimizers:
            opt.step()

