import numpy as np

'''
The game is called Blotto, I hear this puzzle from Jane Street(a quantitative trading company):

There are 10 castles, numbered 1, 2, 3, ... , 10, and worth 1, 2, 3, ... , 10 points respectively.  
You have 100 soldiers, which you can allocate between the castles however you wish.  
Your opponent also (independently) does the same.  
The number of soldiers on each castle is then compared, and for each castle:
· If one player has strictly more than twice as many soldiers on that castle as the other, they win the points for that castle
· Otherwise, it's a tie, and no one gets points for that castle

Here is an example:
Castle	C1	C2	C3	C4	C5	C6	C7	C8	C9	C10
Alice	5	10	10	7	19	1	10	8	20	10
Bob	    2	4	5	20	12	3	4	1	40	9

In this game, Alice wins 1, 2, 7, 8 for 18 points, and Bob wins 4, 6 for 10 points. 
No one wins castle 3, 5, 9, 10.

We use evolution algorithm to solve this question.


'''


def get_fitness(pop, justwin=False, single=False):
    '''
    pred: 10维向量
    '''
    N = 2000  # 希望sample的个数
    fitness = 0
    for i in range(N):
        # sample 一些模拟的布兵可能
        s = sample_soldiers()
        fitness += combat(pop, s, justwin=justwin, single=single)
    return fitness / N


def sample_soldiers():
    sample = np.random.permutation(100)
    sample = sample[0:10]
    sample[-1] = 100
    sample = np.sort(sample)
    sample[1:10] = sample[1:10] - sample[0:9]
    return sample


def combat(p1, p2, justwin=False, single=False):
    """
    player1 & player2 均为10维向量

    如果justwin==True返回胜负结果 (+1,0,-1)
    否则，返回两边相差的分数

    如果single=False 说明p1不是1*10而是N*10
    """
    castles = np.arange(10) + 1  # 第i座城堡值i分
    if single == False:
        score2 = ((2 * p1 < p2) * castles).sum(axis=1)
        score1 = ((p1 > 2 * p2) * castles).sum(axis=1)
    elif single == True:
        score2 = ((2 * p1 < p2) * castles).sum(axis=0)
        score1 = ((p1 > 2 * p2) * castles).sum(axis=0)

    if justwin == True:
        '''
        if score1 > score2: return 1
        if score1 == score2: return 0
        if score1 < score2: return -1
        '''
        return score1 > score2
    else:
        return score1 - score2


def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / fitness.sum())
    return pop[idx]


def toDNA(p):
    '''
    输入10维向量 输出DNA_SIZE=9*7维二进制向量
    '''
    # 把p从10维变为9维的q
    q = np.zeros(9, dtype=int)
    for i in range(9):
        q[i] = int(p[0:i + 1].sum())

    pDNA = np.zeros(DNA_SIZE)
    for i in range(9):
        iDNA = list(map(int, bin(q[i])[2:]))  # 把q的第i个数字转换成二进制
        pDNA[(7 * (i + 1) - len(iDNA)): 7 * (i + 1)] = np.array(iDNA)[:]
    return pDNA


def fromDNA(pDNA):
    '''
    输入9*7维向量 输出10维二进制向量
    '''
    p = np.zeros(10, dtype=int)
    for i in range(9):
        p[i] = pDNA[7 * i: 7 * (i + 1)].dot(2 ** np.arange(7)[::-1])
    p[1:9] = np.sort(p[1:9])
    p[1:9] = p[1:9] - p[0:8]
    p[9] = 100 - p.sum()
    return p


def crossover(parent, p):
    '''
    两个输入都是10维的
    '''
    parentDNA = toDNA(parent)
    pDNA = toDNA(p)
    if np.random.rand() < CROSS_RATE:
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)  # choose crossover points
        parentDNA_copy = parentDNA.copy()
        parentDNA[cross_points] = pDNA[cross_points]  # mating and produce one child
        if (fromDNA(parentDNA) < 0).any():  # 这里就是会出现某一项小于0到情况。。。不知道为啥只能粗暴解决了
            parentDNA = parentDNA_copy
    return fromDNA(parentDNA)


def mutate(p):
    '''
    输入为10维向量
    '''
    pDNA = toDNA(p)
    for i in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            pDNA[i] = 1 if pDNA[i] == 0 else 0
    if (fromDNA(pDNA) < 0).any():
        return p
    else:
        return fromDNA(pDNA)


if __name__ == '__main__':
    POP_SIZE = 100
    N_GENERATIONS = 1000
    DNA_SIZE = 9 * 7  # 100<128=2^7
    CROSS_RATE = 0.8
    MUTATION_RATE = 0.01

    # 自己手动建一个population
    pop = np.array([[0, 5, 12, 4, 11, 15, 12, 14, 13, 14],
                    [5, 0, 4, 8, 10, 11, 17, 16, 14, 15],
                    [1, 0, 3, 8, 10, 10, 11, 23, 13, 21],
                    [0, 3, 14, 6, 8, 11, 17, 13, 10, 18],
                    [0, 6, 9, 7, 14, 11, 12, 15, 14, 12],
                    [1, 3, 6, 15, 9, 9, 15, 12, 19, 11],
                    [0, 3, 5, 7, 10, 10, 19, 13, 15, 18],
                    [0, 3, 8, 8, 11, 13, 13, 18, 11, 15],
                    [1, 2, 3, 10, 10, 12, 13, 21, 15, 13],
                    [1, 2, 3, 10, 10, 12, 13, 21, 15, 13]])

    # initialize the pop (POP_SIZE,10)
    # pop = sample_soldiers()
    for _ in range(POP_SIZE - 10):
        pop = np.vstack((pop, sample_soldiers()))
    # best_ever = sample_soldiers()
    pop_index = 10

    for _ in range(N_GENERATIONS):
        fitness = get_fitness(pop, justwin=True)  # 把justwin设成False就会进化到最坏情况。。。什么鬼？？
        fitness += fitness.min()  # 为了变成正数
        best_present = pop[np.argmax(fitness), :]
        print("找到的最好的布兵: ", best_present, '其fitness为：', get_fitness(best_present, single=True))
        if get_fitness(best_ever, single=True) < get_fitness(best_present, single=True):
            best_ever = best_present

        pop = select(pop, fitness)

        pop_copy = pop.copy()

        for parent in pop:
            # 随机选一个杂交
            i_ = np.random.randint(0, POP_SIZE, size=1)
            child = crossover(parent, pop_copy[i_])
            child = mutate(child)
            parent[:] = child
            if get_fitness(parent, single=True) > 15:
                pop[pop_index, :] = parent[:]
                pop_index += 1
            pop_index %= POP_SIZE

    print('\n全过程中找到的最好布兵：', best_ever, '其fitness为：', get_fitness(best_ever, single=True))