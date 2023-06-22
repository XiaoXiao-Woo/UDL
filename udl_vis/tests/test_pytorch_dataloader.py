import torch
import numpy
import random
from udl_vis.Basis.auxiliary import set_random_seed
print(torch.__version__)  # 1.13.0+cu116

if __name__ == '__main__':

    n = 100
    set_random_seed(1)

    seed = 1#int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    # 不受全局种子限制，设置完后内部仍有种子偏移保证每个epoch都是不同的
    # Function 'torch.Generator()' is not limited by the global seed
    # because there is another seed offset inside it to ensure that each epoch is different
    generator.manual_seed(seed)

    def simple_generator():
        # 随着epoch，torch.randperm内部会产生种子偏移，可以通过generator来控制，注意不可以每轮设成一样的
        # When the epoch is increasing, torch.randperm also generates a seed offset inside,
        # which can be controlled by the generator.
        # Note that it cannot be set the same for each epoch

        # 仅限于load模型的时候在模型、优化器都载入后，保证数据轮次也是一样的
        # When loading the weights, the model and the optimizer are loaded.
        # Besides, the sampled data in dataloader are guaranteed to be the same.
        # See Line: 36-39
        yield torch.randperm(n, generator=generator).tolist()


    state = generator.get_state()

    for i in range(3):
        my_gen = simple_generator()
        # 使得每轮epoch是一样的，但不要这么做，仅限于恢复模型的时候保证数据轮次是一样的
        # It will be make sure each epoch is the same, but don't do this,
        # only when restoring the model to ensure that the sampled data are the same.
        # It should be considered in https://pytorch.org/docs/stable/notes/randomness.html
        print(generator.set_state(state))
        print(generator.get_state().sum())
        # print(torch.get_rng_state().sum())
        print(next(my_gen))

    # 跟实例相关的，不是全局的
    # It is related to the instance, not the global.
    print(torch.Generator().get_state().sum())
