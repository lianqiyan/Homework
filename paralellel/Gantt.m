clear;  
axis([0,56,0,6.5]);%x轴 y轴的范围
set(gca,'xtick',0:2:56) ;%x轴的增长幅度
set(gca,'ytick',0:1:6.5) ;%y轴的增长幅度
xlabel('加工时刻'),ylabel('机器号');%x轴 y轴的名称
title('MT06 的一个最佳调度（最短完工时间55）');%图形的标题
n_bay_nb=6;%total bays  //机器数目
n_task_nb = 36;%total tasks  //任务数目
%x轴 对应于画图位置的起始坐标x
n_start_time=[0 1 0 8 8 13 6 10 13 1 16 13 22 13 22 25 18 27 25 19 28 31 28 31 38 30 38 42 45 38 48 45 49 49 48 52];%start time of every task  //每个工序的开始时间
%length 对应于每个图形在x轴方向的长度
n_duration_time =[1 5 8 5 5 3 4 8 10 3 3 9 3 5 5 6 9 3 5 9 10 1 10 7 4 8 10 3 9 7 3 4 6 1 4 1];%duration time of every task  //每个工序的持续时间
%y轴 对应于画图位置的起始坐标y
n_bay_start=[2 2 1 1 2 1 3 5 4 0 3 2 1 0 2 1 0 3 4 5 0 1 5 3 5 4 0 5 5 4 0 4 4 2 3 3]; %bay id of every task  ==工序数目，即在哪一行画线
%工序号，可以根据工序号选择使用哪一种颜色
n_job_id=[0 2 1 3 1 5 2 2 1 0 5 4 4 3 3 0 2 3 4 5 5 2 1 0 4 3 1 0 3 2 4 5 0 5 1 4];%
rec=[0,0,0,0];%temp data space for every rectangle  
color=['r','g','b','c','m','y'];
for i =1:n_task_nb  
  rec(1) = n_start_time(i);%矩形的横坐标
  rec(2) = n_bay_start(i)+0.7;  %矩形的纵坐标
  rec(3) = n_duration_time(i);  %矩形的x轴方向的长度
  rec(4) = 0.6; 
  txt=sprintf('J%d', n_job_id(i)+1);%将机器号，工序号，加工时间连城字符串
   rectangle('Position',rec,'LineWidth',0.5,'LineStyle','-');%draw every rectangle  
   text(n_start_time(i)+0.2,(n_bay_start(i)+1),txt,'FontWeight','Bold','FontSize',18);%label the id of every task  ，字体的坐标和其它特性
end