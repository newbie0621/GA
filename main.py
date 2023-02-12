import numpy as np
import matplotlib.pyplot as plt


###将十进制转换为二进制
def tenTotwo(number):
    # 定义栈
    s = []
    binstring = ''
    while number > 0:
        # 余数进栈
        rem = number % 2
        s.append(rem)
        number = number // 2
    while len(s) > 0:
        # 元素全部出栈即为所求二进制数
        binstring = binstring + str(s.pop())
    # 这里不管是个位，十分位还是百分位都应该是四位数，不够的用0填充
    if len(binstring) < 4:
        binstring = '0' * (4 - len(binstring)) + binstring
    return binstring


### 种群初始化
def initialize(P0):
    initial_population = []
    for i in range(P0):
        # 生成个位数、十分位数和百分位数
        a = np.random.randint(low=1, high=4)
        b = np.random.randint(low=0, high=10)
        c = np.random.randint(low=0, high=10)
        initial_population.append(encoder(a, b, c))
    return initial_population


### 编码
def encoder(a, b, c):
    # 将个位、十分位、百分位转换为二进制再拼接
    return tenTotwo(a) + tenTotwo(b) + tenTotwo(c)


### 解码(这里的s是一个字符串)
def decoder(s):
    a = int(s[:4], 2)
    b = int(s[4:8], 2)
    c = int(s[8:], 2)
    return a + b * 0.1 + c * 0.01


### 交叉操作
def cross(pop, pc, P):
    # 这里的P代表交叉操作后种群的规模大小
    new_pop = []
    while len(new_pop) < P:
        for i, father in enumerate(pop):
            if np.random.uniform() < pc:
                # 如果小于概率pc,则应该在种群中再找到另外一个个体作为母亲执行交叉操作(确保父亲母亲是不同个体)
                while True:
                    index = np.random.randint(low=0, high=len(pop))
                    if index != i:
                        break
                mother = pop[index]
                # 字符串本身不能进行赋值，所以要转换为列表
                father = list(father)
                mother = list(mother)
                # 生成交叉点
                cross_points = np.random.randint(low=1, high=len(father) - 1)
                child_1 = father[:cross_points] + mother[cross_points:]
                child_2 = mother[:cross_points] + father[cross_points:]
                child_1 = ''.join(child_1)
                child_2 = ''.join(child_2)
                if (decoder(child_1) >= 1) and (decoder(child_1) <= 4):
                    new_pop.append(child_1)
                if (decoder(child_2) >= 1) and (decoder(child_2) <= 4):
                    new_pop.append(child_2)
            else:
                new_pop.append(father)
            if len(new_pop) == P:
                break
    return new_pop


### 变异操作
def mutation(pop, pm):
    new_pop = []
    for child in pop:
        if np.random.uniform() < pm:
            # 如果小于概率pm则要对该个体执行变异操作
            mutate_point = np.random.randint(low=0, high=len(child))
            # 字符串本身不能进行赋值，所以要转换为列表
            child_ = list(child)
            # 所谓变异，即将原来的1变异为0,0变异为1
            if child_[mutate_point] == '0':
                child_[mutate_point] = '1'
            else:
                child_[mutate_point] = '0'
            child_ = ''.join(child_)
            # 判断变异后的个体是否超出了范围
            if decoder(child_) >= 1 and decoder(child_) <= 4:
                child = child_
        new_pop.append(child)
    return new_pop


### 选择操作(用到轮盘赌)
def selection(pop, fitness, P0):
    new_pop = []
    # 求总的适应度的和
    sum_fits = np.sum(fitness)
    # 计算累积概率
    cumulative_probability = []
    for i in range(len(fitness)):
        cumulative_probability.append(np.sum(fitness[:i + 1]) / sum_fits)
    # 选择优良个体
    for i in range(P0):
        u = np.random.uniform()
        # 判断u掉进哪个区间
        for j in range(len(cumulative_probability)):
            if cumulative_probability[j] >= u:
                new_pop.append(pop[j])
                # 当找到所在区间后要跳出循环
                break
    return new_pop


###计算适应度(这里的s是一个12位的字符串)
def get_fitness(s):
    '''
        f=(x-2)^2,因为是求最小值，所以等价于求1/(x-2)^2最大值
        加上1e-5是防止分母为零，对结果影响非常小

    '''
    x = decoder(s)

    return 1.0 / ((x - 2) ** 2 + 1e-5)


### 找出种群中最优个体
def select_best(pop):
    fitness_list = []
    individual_list = []
    # 计算每个个体解码后对应的值和其对应的适应度（函数值）
    for individual in pop:
        individual_list.append(decoder(individual))
        fitness_list.append(get_fitness(individual))
    # 寻找适应度最大的个体的下标
    index = fitness_list.index(max(fitness_list))
    # 注意这里的适应度需要还原
    return round(individual_list[index], 2), round(1.0 / fitness_list[index], 2)


if __name__ == '__main__':
    fitness_history = []
    x_history = []
    Gen = [10, 20, 20, 20, 20, 20]
    pc = [0.6, 0.6, 0.6, 0.2, 0.6, 0.6]
    pm = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    P0 = [10, 10, 10, 10, 20, 5]
    P = [15, 15, 15, 15, 30, 7]
    for i in range(6):
        x_list = []
        fitness_list = []
        # 种群初始化
        pop = initialize(P0[i])
        for j in range(Gen[i]):
            # 交叉操作
            pop = cross(pop, pc[i], P[i])
            # 变异操作
            pop = mutation(pop, pm[i])
            # 选择操作
            fitness = [get_fitness(s) for s in pop]
            pop = selection(pop, fitness, P0[i])
            x_list.append(select_best(pop)[0])
            fitness_list.append(select_best(pop)[1])
        x_history.append(x_list)
        fitness_history.append(fitness_list)

    ###绘图
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.plot(x_history[i], label='x')
        plt.plot(fitness_history[i], label='fitness')
        plt.xlabel('epoch')
        plt.ylabel('fitness')
        plt.legend(loc='lower right')
        plt.title('Gen={},pc={},pm={},P0={},P={}'.format(Gen[i], pc[i], pm[i], P0[i], P[i]))
    plt.tight_layout()
    plt.show(block=True)