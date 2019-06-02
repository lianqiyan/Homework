clear;  
axis([0,56,0,6.5]);%x�� y��ķ�Χ
set(gca,'xtick',0:2:56) ;%x�����������
set(gca,'ytick',0:1:6.5) ;%y�����������
xlabel('�ӹ�ʱ��'),ylabel('������');%x�� y�������
title('MT06 ��һ����ѵ��ȣ�����깤ʱ��55��');%ͼ�εı���
n_bay_nb=6;%total bays  //������Ŀ
n_task_nb = 36;%total tasks  //������Ŀ
%x�� ��Ӧ�ڻ�ͼλ�õ���ʼ����x
n_start_time=[0 1 0 8 8 13 6 10 13 1 16 13 22 13 22 25 18 27 25 19 28 31 28 31 38 30 38 42 45 38 48 45 49 49 48 52];%start time of every task  //ÿ������Ŀ�ʼʱ��
%length ��Ӧ��ÿ��ͼ����x�᷽��ĳ���
n_duration_time =[1 5 8 5 5 3 4 8 10 3 3 9 3 5 5 6 9 3 5 9 10 1 10 7 4 8 10 3 9 7 3 4 6 1 4 1];%duration time of every task  //ÿ������ĳ���ʱ��
%y�� ��Ӧ�ڻ�ͼλ�õ���ʼ����y
n_bay_start=[2 2 1 1 2 1 3 5 4 0 3 2 1 0 2 1 0 3 4 5 0 1 5 3 5 4 0 5 5 4 0 4 4 2 3 3]; %bay id of every task  ==������Ŀ��������һ�л���
%����ţ����Ը��ݹ����ѡ��ʹ����һ����ɫ
n_job_id=[0 2 1 3 1 5 2 2 1 0 5 4 4 3 3 0 2 3 4 5 5 2 1 0 4 3 1 0 3 2 4 5 0 5 1 4];%
rec=[0,0,0,0];%temp data space for every rectangle  
color=['r','g','b','c','m','y'];
for i =1:n_task_nb  
  rec(1) = n_start_time(i);%���εĺ�����
  rec(2) = n_bay_start(i)+0.7;  %���ε�������
  rec(3) = n_duration_time(i);  %���ε�x�᷽��ĳ���
  rec(4) = 0.6; 
  txt=sprintf('J%d', n_job_id(i)+1);%�������ţ�����ţ��ӹ�ʱ�������ַ���
   rectangle('Position',rec,'LineWidth',0.5,'LineStyle','-');%draw every rectangle  
   text(n_start_time(i)+0.2,(n_bay_start(i)+1),txt,'FontWeight','Bold','FontSize',18);%label the id of every task  ��������������������
end