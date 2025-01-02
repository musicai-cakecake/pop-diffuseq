import torch
import numpy


def lm_head_1(in_tensor):
    return in_tensor+0.1


def lm_head_2(in_tensor):
    return in_tensor+0.2


def lm_head_3(in_tensor):
    return in_tensor+0.3


def lm_head_4(in_tensor):
    return in_tensor+0.4


a = numpy.arange(16384).reshape((4, 512, 8))

a_torch = torch.tensor(a)

print('a_torch.shape', a_torch.shape)
print(a_torch[0][0])
print(a_torch[0][1])
print(a_torch[0][2])
print(a_torch[0][3])
print(a_torch[0][4])
print(a_torch[0][5])
print(a_torch[0][6])
print(a_torch[0][7])
print(a_torch[0][8])
print(a_torch[0][9])
print(a_torch[0][10])
print(a_torch[0][11])
print(a_torch[0][12])
print(a_torch[0][13])
print(a_torch[0][14])
print(a_torch[0][15])
print('a_torch[3][501]: ', a_torch[3][501])
print('a_torch[3][502]: ', a_torch[3][502])
print('a_torch[3][503]: ', a_torch[3][503])
print('a_torch[3][504]: ', a_torch[3][504])
print('a_torch[3][505]: ', a_torch[3][505])
print('a_torch[3][506]: ', a_torch[3][506])
print('a_torch[3][507]: ', a_torch[3][507])
print('a_torch[3][508]: ', a_torch[3][508])
print('a_torch[3][509]: ', a_torch[3][509])
print('a_torch[3][510]: ', a_torch[3][510])
print('a_torch[3][511]: ', a_torch[3][511])

lm_src = lm_head_1(a_torch[:, 0:11, :])
lm_5 = lm_head_1(a_torch[:, 11:508:4, :])

print("\nlm_5[-1, -1]:", lm_5[-1, -1])
print("\nlm_5[-1, 0]:", lm_5[-1, 0])
print("\nlm_5[-1, 1]:", lm_5[-1, 1])

lm_head_1_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])

lm_5 = lm_5 * lm_head_1_mask

print("\nlm_5 mask [-1, -1]:", lm_5[-1, -1])
print("\nlm_5 mask [-1, 0]:", lm_5[-1, 0])
print("\nlm_5 mask [-1, 1]:", lm_5[-1, 1])

lm_6 = lm_head_2(a_torch[:, 12:509:4, :])
print("\nlm_6:", lm_6[-1, -1])
lm_7 = lm_head_3(a_torch[:, 13:510:4, :])

lm_8 = lm_head_4(a_torch[:, 14:511:4, :])
lm_end = lm_head_1(a_torch[:, 511, :]).unsqueeze(1)

# Multiply by zero on the irrelevant categories, Multiply one for the target category.

lm_stack_trg = torch.stack((lm_5, lm_6, lm_7, lm_8), dim=2)
lm_trg = lm_stack_trg.view([lm_stack_trg.shape[0], -1, lm_stack_trg.shape[-1]])
lm_out = torch.cat((lm_src, lm_trg, lm_end), dim=1)

# lm_src = self.lm_head_1(hidden_repr[:, 0:10, :])
# lm_5 = self.lm_head_1(hidden_repr[:, 10:508:4, :])
# lm_6 = self.lm_head_2(hidden_repr[:, 11:509:4, :])
# lm_7 = self.lm_head_3(hidden_repr[:, 12:510:4, :])
# lm_8 = self.lm_head_4(hidden_repr[:, 13:511:4, :])
# lm_end = self.lm_head_1(hidden_repr[:, 511, :]).unsqueeze(1)
# lm_stack_trg = torch.stack((lm_5, lm_6, lm_7, lm_8), dim=2)
# lm_trg = lm_stack_trg.view([lm_stack_trg.shape[0], -1, lm_stack_trg.shape[-1]])
# lm_out = torch.cat((lm_src, lm_trg, lm_end), dim=1)

print('lm_out.shape: ', lm_out.shape)
print(lm_out[0][0])
print(lm_out[0][1])
print(lm_out[0][2])
print(lm_out[0][3])
print(lm_out[0][4])
print(lm_out[0][5])
print(lm_out[0][6])
print(lm_out[0][7])
print(lm_out[0][8])
print(lm_out[0][9])
print(lm_out[0][10])
print(lm_out[0][11])
print(lm_out[0][12])
print(lm_out[0][13])
print(lm_out[0][14])
print(lm_out[0][15])
print(lm_out[0][16])
print(lm_out[0][17])
print(lm_out[0][18])
print(lm_out[0][19])
print(lm_out[0][20])
print('lm_out[3][507]: ', lm_out[3][507])
print('lm_out[3][508]: ', lm_out[3][508])
print('lm_out[3][509]: ', lm_out[3][509])
print('lm_out[3][510]: ', lm_out[3][510])
print('lm_out[3][511]: ', lm_out[3][511])
