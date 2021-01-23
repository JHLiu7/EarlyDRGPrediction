import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable


class Base(nn.Module):
    def __init__(self, args, embedding):
        super(Base, self).__init__()

        self.embed_num, self.embed_size = embedding.shape
        weight = torch.from_numpy(embedding).float()
        self.embed = nn.Embedding(embedding_dim=self.embed_size, num_embeddings=self.embed_num, padding_idx=0, _weight=weight)
        self.embed.weight.requires_grad=True

        self.dropout = nn.Dropout(args.dropout_rate)

        self.target = args.target

        # final layers to produce real number ouput
        self.w_layer = nn.Linear(args.Y, 100)
        self.outlayer= nn.Linear(100, 1)

    def _compute_loss(self, yhat, target):
        if self.target == 'drg':
            logit= yhat
            loss = F.cross_entropy(logit, target)
        elif self.target == 'rw':
            yhat = self.w_layer(yhat)
            yhat = F.relu(yhat)
            yhat = self.outlayer(yhat)
            logit = yhat.squeeze()
            
            loss = F.mse_loss(logit, target)
    
        return logit, loss

class CAML(Base):
    def __init__(self, args, embedding):
        super(CAML, self).__init__(args, embedding)


        self.conv = nn.Conv1d(self.embed_size, args.cnn_filter_maps, kernel_size=args.single_kernel_size, padding=int(math.floor(args.single_kernel_size/2)))
        nn.init.xavier_uniform_(self.conv.weight)

        self.U = nn.Linear(args.cnn_filter_maps, args.Y)
        self.fc= nn.Linear(args.cnn_filter_maps, args.Y)

        nn.init.xavier_uniform_( self.U.weight)
        nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x, target):
        x = self.embed(x) # (N, W, D)
        x = self.dropout(x)
        x = x.transpose(1,2) # (N, D, W)

        x = F.relu(self.conv(x)) # (N, F, W)

        alpha = F.softmax(self.U.weight.matmul(x), dim=2) # (N, Y, W); U.weight->(Y, F)

        m = alpha.matmul(x.transpose(1,2)) # (N, Y, F)
        m = self.fc.weight.mul(m) # (N, Y, F) element-wise, final.w -> (Y,F)
        
        m = m.sum(dim=2).add(self.fc.bias) # (N, Y)
        logit, loss = self._compute_loss(m, target)

        return logit, loss

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, use_res, dropout):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=int(math.floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel),
            nn.Tanh(),
            nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=int(math.floor(kernel_size / 2)), bias=False),
            nn.BatchNorm1d(outchannel)
        )

        self.use_res = use_res
        if self.use_res:
            self.shortcut = nn.Sequential(
                        nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm1d(outchannel)
                    )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.left(x)
        if self.use_res:
            out += self.shortcut(x)
        out = torch.tanh(out)
        out = self.dropout(out)
        return out

class MultiResCNN(Base):
    def __init__(self, args, embedding):
        super(MultiResCNN, self).__init__(args, embedding)

        self.conv_dict = {1: [self.embed_size, args.cnn_filter_maps],
                     2: [self.embed_size, 100, args.cnn_filter_maps],
                     3: [self.embed_size, 150, 100, args.cnn_filter_maps],
                     4: [self.embed_size, 200, 150, 100, args.cnn_filter_maps]
                     }
        conv_layers=1 if not args.conv_layers else args.conv_layers
        conv_dim = self.conv_dict[conv_layers]

        self.convs = nn.ModuleList()
        kernels = args.multi_kernel_sizes.split(',')
        # for i in range(conv_layers):
        for k in kernels:
            one_channel = nn.ModuleList()
            cnn = nn.Conv1d(self.embed_size, self.embed_size, kernel_size=int(k),
                padding=int(math.floor(int(k)/2)) )
            nn.init.xavier_uniform_(cnn.weight)
            one_channel.add_module('baseconv', cnn)

            # res = ResidualBlock(self.embed_size, args.cnn_filter_maps, int(k), 1, True, args.dropout_rate)
            # one_channel.add_module('resconv', res)
            for i in range(conv_layers):
                res = ResidualBlock(conv_dim[i], conv_dim[i+1], int(k), 1, True, args.dropout_rate)
                one_channel.add_module('resconv-{}'.format(i), res)

            self.convs.add_module('channel-{}'.format(k), one_channel)

        hsize = args.cnn_filter_maps * len(kernels)

        self.U = nn.Linear(hsize, args.Y)
        self.fc= nn.Linear(hsize, args.Y)
        nn.init.xavier_uniform_( self.U.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        try:
            if args.no_attn_pooling:
                self.no_attn_pooling = True 
        except:
            self.no_attn_pooling = False

    def forward(self, x, target):
        # features = [self.embed(x)]
        # x = torch.cat(features, dim=2)
        x = self.embed(x)
        x = self.dropout(x)
        x = x.transpose(1,2) # (N, D, W)

        conv_result = []
        for conv in self.convs:
            tmp = x
            for idx, md in enumerate(conv):
                if idx == 0:
                    tmp = torch.tanh(md(tmp))
                else:
                    tmp = md(tmp)
            tmp = tmp.transpose(1, 2)  # (N, W, F)
            conv_result.append(tmp)


        if not self.no_attn_pooling:
            x = torch.cat(conv_result, dim=2) # (N, W, nF)

            alpha = F.softmax(self.U.weight.matmul(x.transpose(1,2)), dim=2)
            m = alpha.matmul(x) 
            m = self.fc.weight.mul(m).sum(dim=2).add(self.fc.bias)
        else:
            x = [m.transpose(1,2) for m in conv_result] # [(N, F, W)]
            x = [F.max_pool1d(m, m.size(-1)).squeeze(-1) for m in x]
            x = torch.cat(x, dim=1) # (N, nF)

            m = self.fc(self.dropout(x))

        logit, loss = self._compute_loss(m, target)
        return logit, loss

class LSTM4Struct(nn.Module):
    def __init__(self, args, ouput_last=True):
        super(LSTM4Struct, self).__init__()

        Y = args.Y
        hidden_size = args.hidden_size
        dropout = args.dropout

        self.rnn = nn.LSTM(input_size=208, hidden_size=hidden_size, batch_first=True)

        self.U = nn.Linear(hidden_size, Y)
        self.fc= nn.Linear(hidden_size, Y)
        # self.fc2= nn.Linear(hidden_size, Y)

        nn.init.xavier_uniform_( self.U.weight)
        nn.init.xavier_uniform_(self.fc.weight)

        self.ouput_last = ouput_last
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, target):
        h, (h_n, c_n) = self.rnn(x) # h (N, step, H), h_n (1, N, H)

        if self.ouput_last:
            logit = self.fc(h_n.squeeze())
        else:
            h = self.dropout(h)
            # apply attention
            alpha = F.softmax(self.U.weight.matmul(h.transpose(1,2)), dim=2)
            m = alpha.matmul(h)
            logit = self.fc.weight.mul(m).sum(dim=2).add(self.fc.bias)

        loss = F.cross_entropy(logit, target)

        return logit, loss

def pick_model(args, embedding):

    if args.model == "CAML":
        model = CAML(args, embedding)
    elif args.model == "MultiResCNN":
        model = MultiResCNN(args, embedding)
    elif args.model == "MultiResCNN_no_attn":
        args.no_attn_pooling = True
        model = MultiResCNN(args, embedding)
    else:
        model = None
    return model

# def pick_model(args, embedding):
#     if args.model == "MultiCNN":
#         model = models.MultiCNN(args, embedding)
#     elif args.model == "CAML":
#         model = models.CAML(args, embedding)
#     elif args.model == "MultiResCNN":
#         model = models.MultiResCNN(args, embedding)
#     elif args.model == "MultiResCNN_no_attn":
#         args.no_attn_pooling = True
#         model = models.MultiResCNN(args, embedding)
#     else:
#         model = None
#     return model