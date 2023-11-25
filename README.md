# GA_

## 一、

### test_plot_ ifft_uv.py是原模型。
### theta_r和phi_r（43-44行）分别是接收器的仰角和方位角，可修改括号中的数字（角度）。仰角和方位角范围分别为[0, 60°]和[0, 360°]。（参考论文第2页左侧图）
### 输出的第一个图是RIS平面控制的晶元状态的矩阵（1-bit只有0和 $\pi$ 俩个状态），第二个图是RIS平面的2d辐射方向图。
### 原模型的主要想法是通过控制RIS平面的各个晶元状态，来增强指定接收器方向的信号。


## 二、

### 255-298行是论文第3页右下角的GA目标函数， $M_L(u,v)$ 和 $M_U(u,v)$ 定义在第4页左侧。
### **First objective function**是公式第一个求和，**Second**是第二个。
### values是公式中的 $\hat{E}(u,v)$ ，E_L是 $E_L(u,v;\Theta_{hp})$ 。
### 另外，公式中| · |符号是代表一个集合中元素的个数。 $(u,v)$ 是位置的角坐标。


## 三、代码逻辑（255-298）:

$$\begin{aligned}F_{obj}^2 & =\sum_{(u,v)\in S_L}\left(\frac{\hat E(u,v)-M_L(u,v)}{|S_L|}\right)^2 \\  & +\sum_{(u,v)\in S_U}\left(\frac{\hat E(u,v)-M_U(u,v)}{|S_U|}\right)^2\end{aligned}$$

$$\mathcal{E}(u,v;\Theta)=(u-u_0)^2+(v-v_0)^2-\sin^2(\Theta)$$



### 1. 先对所有 $(u,v)$ 计算 $M_L(u,v)$ 和 $M_U(u,v)$ ；
### 2. 再分别遍历 $M_L(u,v)$ 和 $M_U(u,v)$ ，符合 $S_L$ 和 $S_U$ 条件的，分别计算 $(\hat{E}(u,v)-M_L(u,v))^2$ 和 $(\hat{E}(u,v)-M_U(u,v))^2$ 丢进去列表F_1和F_2。
### 3. 然后，F_1_sum是F_1的和除以它的个数平方，F_2_sum同样的操作。
### 4. 最后，输出的F_obj是俩相加，也就是论文的目标函数。


## 四、GA优化想法

### 目标：增大主光束的半功率波束宽度 (HPBW)，减小旁瓣电平 (SLL)   →→→转化为→→→   最小化目标函数
### 输入：随机二进制矩阵（代表1-bit模式下的晶元状态）
### 输出：最优矩阵
### 初始种群：随机二进制矩阵
### 采样算子：二元随机采样（Binary random Sampling）
### 选择算子：默认二元锦标赛选择（Binary tournament Selection）
### 交叉算子：两点交叉（Two-point Crossover）
### 变异算子：位翻转变异（Bit flip Mutation）



