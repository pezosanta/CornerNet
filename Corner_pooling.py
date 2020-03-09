import torch
import torch.nn as nn
from utils import convolution

def comp(a,b,A,B):
    batch = a.size(0)
    a_ = a.unsqueeze(1).contiguous().view(batch,1,-1)
    b_ = b.unsqueeze(1).contiguous().view(batch,1,-1)
    c_ = torch.cat((a_,b_),1)
    m = c_.max(1)[0].unsqueeze(1).expand_as(c_)
    m = (c_==m).float()
    m1 = m.permute(0,2,1)
    k = m1[...,0]
    j = m1[...,1]
    z = ((k*j)!=1).float()
    j = z*j
    m1 = torch.cat((k,j),1).unsqueeze(1).view_as(m)

    A_ = A.unsqueeze(1).contiguous().view(batch,1,-1)
    B_ = B.unsqueeze(1).contiguous().view(batch,1,-1)
    C_ = torch.cat((A_,B_),1).permute(0,2,1)
    m1 = m1.long().permute(0,2,1)
    res = C_[m1.long()==1].view_as(a)

    return res

class left_pool_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_.clone())
        output = torch.zeros_like(input_)
        batch = input_.size(0)
        width = input_.size(3)
        
        input_tmp = input_.index_select(3, torch.tensor([width-1]).cuda())
        #print("input_tmp shape:{}, input_tmp:{} ".format(input_tmp.shape, input_tmp))
        output.index_select(3,torch.tensor([width-1]).cuda()).copy_(input_tmp)
        #print("output_tmp shape:{}, output_tmp:{} ".format(output.shape, output))
        
        for idx in range(1, width):
            input_tmp = input_.index_select(3,torch.tensor([width-idx-1]).cuda())
            #print("i:{}, input_tmp_shape: {}, input_tmp:{}".format(idx, input_tmp.shape, input_tmp))
            output_tmp = output.index_select(3,torch.tensor([width-idx]).cuda())
            #print("i:{}, output_tmp_shape: {}, output_tmp:{}".format(idx, output_tmp.shape, output_tmp))
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1)
            #print("i:{}, cmp_tmp_shape: {}, cmp_tmp:{}".format(idx, cmp_tmp.shape, cmp_tmp))
            cmp_tmp_max = cmp_tmp.max(1)[0]
            #print("i:{}, cmp_tmp_max_shape: {}, cmp_tmp_max:{}".format(idx, cmp_tmp_max.shape, cmp_tmp_max))
            output.index_select(3,torch.tensor([width-idx-1]).cuda()).copy_(cmp_tmp_max.view_as(input_tmp))
            #print("i:{}, output_shape: {}, output:{}".format(idx, output.shape, output))
         
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        output = torch.zeros_like(input_)
        
        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)
        
        w = input_.size(3)
        batch = input_.size(0)
        
        output_tmp = res.index_select(3, torch.tensor([w-1]).cuda())
        grad_output_tmp = grad_output.index_select(3, torch.tensor([w-1]).cuda())
        output_tmp.copy_(grad_output_tmp)
        
        input_tmp = input_.index_select(3, torch.tensor([w-1]).cuda())
        output.index_select(3,torch.tensor([w-1]).cuda()).copy_(input_tmp)
        
        for idx in range(1, w):
            
            input_tmp = input_.index_select(3, torch.tensor([w-idx-1]).cuda())
            output_tmp = output.index_select(3,torch.tensor([w-idx]).cuda())
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.index_select(3,torch.tensor([w-idx-1]).cuda()).copy_(cmp_tmp.view_as(input_tmp))
            
            grad_output_tmp = grad_output.index_select(3, torch.tensor([w-idx-1]).cuda())
            res_tmp = res.index_select(3,torch.tensor([w-idx]).cuda())
            com_tmp = comp(input_tmp,output_tmp,grad_output_tmp,res_tmp)
            res.index_select(3,torch.tensor([w-idx-1]).cuda()).copy_(com_tmp)
        return res

class top_pool_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = torch.zeros_like(input_)
        height = output.size(2)
        batch = input_.size(0)

        input_tmp = input_.select(2, height-1)
        output.select(2,height-1).copy_(input_tmp)
        
        for idx in range(1, height):
            input_tmp = input_.select(2, height-idx-1)
            output_tmp = output.select(2,height-idx)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]

            output.select(2, height-idx-1).copy_(cmp_tmp.view_as(input_tmp))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        output = torch.zeros_like(input_)
        
        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)
        
        height = output.size(2)
        batch = input_.size(0)
        #copy the last row
        input_tmp = input_.select(2, height-1)
        output.select(2,height-1).copy_(input_tmp)
        
        grad_tmp = grad_output.select(2, height-1)
        res.select(2,height-1).copy_(grad_tmp)
        for idx in range(1, height):
            input_tmp = input_.select(2, height-idx-1)
            output_tmp = output.select(2,height-idx)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.select(2, height-idx-1).copy_(cmp_tmp.view_as(input_tmp))
            
            grad_output_tmp = grad_output.select(2, height-idx-1)
            res_tmp = res.select(2,height-idx)
            
            com_tmp = comp(input_tmp,output_tmp,grad_output_tmp,res_tmp)
            res.select(2,height-idx-1).copy_(com_tmp)
        return res

class right_pool_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        
        output = torch.zeros_like(input_)
        width = input_.size(3)
        batch = input_.size(0)
 
        input_tmp = input_.select(3, 0)
        output.select(3,0).copy_(input_tmp)
        
        for idx in range(1, width):
            input_tmp = input_.select(3,idx)
            output_tmp = output.select(3,idx-1)

            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.select(3,idx).copy_(cmp_tmp.view_as(input_tmp))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        output = torch.zeros_like(input_)
        
        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)
        
        w = input_.size(3)
        batch = input_.size(0)
        
        output_tmp = res.select(3, 0)
        grad_output_tmp = grad_output.select(3, 0)
        output_tmp.copy_(grad_output_tmp)
        
        input_tmp = input_.select(3, 0)
        output.select(3,0).copy_(input_tmp)
        
        for idx in range(1, w):
            input_tmp = input_.select(3,idx)
            output_tmp = output.select(3,idx-1)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.select(3,idx).copy_(cmp_tmp.view_as(input_tmp))
            
            grad_output_tmp = grad_output.select(3, idx)
            res_tmp = res.select(3,idx-1)
            com_tmp = comp(input_tmp,output_tmp,grad_output_tmp,res_tmp)
            res.select(3,idx).copy_(com_tmp)
        return res

class bottom_pool_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = torch.zeros_like(input_)
        height = output.size(2)
        batch = output.size(0)

        input_tmp = input_.select(2, 0)
        output.select(2,0).copy_(input_tmp)
        
        for idx in range(1, height):
            input_tmp = input_.select(2, idx)
            output_tmp = output.select(2,idx-1)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.select(2, idx).copy_(cmp_tmp.view_as(input_tmp))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        output = torch.zeros_like(input_)
        
        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)
        
        height = output.size(2)
        batch = output.size(0)
 
        input_tmp = input_.select(2,0)
        output.select(2,0).copy_(input_tmp)
        
        grad_tmp = grad_output.select(2,0)
        res.select(2,0).copy_(grad_tmp)
        
        for idx in range(1, height):
            input_tmp = input_.select(2, idx)
            output_tmp = output.select(2,idx-1)
            cmp_tmp = torch.cat((input_tmp.contiguous().view(batch,1,-1),output_tmp.contiguous().view(batch,1,-1)),1).max(1)[0]
            output.select(2, idx).copy_(cmp_tmp.view_as(input_tmp))
            
            grad_output_tmp = grad_output.select(2, idx)
            res_tmp = res.select(2,idx-1)
            
            com_tmp = comp(input_tmp,output_tmp,grad_output_tmp,res_tmp)
            res.select(2,idx).copy_(com_tmp)
        return res

class CornerPool_module(nn.Module):
    def __init__(self):
        super(CornerPool_module, self).__init__()

        self.conv3_bn_relu_in_T = convolution(k = 3, inp_dim = 256, out_dim = 128, stride = 1, with_bn = True)
        self.conv3_bn_relu_in_L = convolution(k = 3, inp_dim = 256, out_dim = 128, stride = 1, with_bn = True)
        self.conv3_bn_relu_in_B = convolution(k = 3, inp_dim = 256, out_dim = 128, stride = 1, with_bn = True)
        self.conv3_bn_relu_in_R = convolution(k = 3, inp_dim = 256, out_dim = 128, stride = 1, with_bn = True)

        self.conv3_bn_TL = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), padding = (1, 1), bias = False),
            nn.BatchNorm2d(256)
        )
        self.conv3_bn_BR = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), padding = (1, 1), bias = False),
            nn.BatchNorm2d(256)
        )

        self.conv1_bn_TL = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1, 1), bias = False),
            nn.BatchNorm2d(256)
        )
        self.conv1_bn_BR = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1, 1), bias = False),
            nn.BatchNorm2d(256)
        )

        self.relu_TL = nn.ReLU(inplace = True)
        self.relu_BR = nn.ReLU(inplace = True)

        self.conv3_bn_relu_out_TL = convolution(k = 3, inp_dim = 256, out_dim = 256, stride = 1, with_bn = True)
        self.conv3_bn_relu_out_BR = convolution(k = 3, inp_dim = 256, out_dim = 256, stride = 1, with_bn = True)

        #self.top_pool = top_pool()
        #self.left_pool = left_pool()
        #self.bottom_pool = bottom_pool()
        #self.right_pool = right_pool()

    def forward(self, x):
        
        # Top-Left pooling
        conv3_bn_relu_in_T = self.conv3_bn_relu_in_T(x)
        top_pool = top_pool_func.apply(conv3_bn_relu_in_T)

        conv3_bn_relu_in_L = self.conv3_bn_relu_in_L(x)
        left_pool = left_pool_func.apply(conv3_bn_relu_in_L)

        conv3_bn_TL = self.conv3_bn_TL(top_pool + left_pool)

        conv1_bn_TL = self.conv1_bn_TL(x)

        relu_TL = self.relu_TL(conv3_bn_TL + conv1_bn_TL)

        conv3_bn_relu_out_TL = self.conv3_bn_relu_out_TL(relu_TL)

        # Bottom-Right pooling
        conv3_bn_relu_in_B = self.conv3_bn_relu_in_B(x)
        bottom_pool = bottom_pool_func.apply(conv3_bn_relu_in_B)

        conv3_bn_relu_in_R = self.conv3_bn_relu_in_R(x)
        right_pool = right_pool_func.apply(conv3_bn_relu_in_R)

        conv3_bn_BR = self.conv3_bn_BR(bottom_pool + right_pool)

        conv1_bn_BR = self.conv1_bn_BR(x)

        relu_BR = self.relu_BR(conv3_bn_BR + conv1_bn_BR)

        conv3_bn_relu_out_BR = self.conv3_bn_relu_out_BR(relu_BR)

        return conv3_bn_relu_out_TL, conv3_bn_relu_out_BR