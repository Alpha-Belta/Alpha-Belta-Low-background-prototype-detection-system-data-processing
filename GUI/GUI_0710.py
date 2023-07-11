# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 17:03:33 2023

@author: Yuanfei Cheng
"""

import socket
import time
from tkinter import * 
from tkinter import ttk,filedialog,messagebox
from numpy import mean,std
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from datetime import datetime

take_packet_header = 0x5a #channel c packet header  数据通道（通道c）
take_SOE_bit = 0x40 #start of event packet
take_EOE_bit = 0x20 #end of event packet
take_high3bits = 0xe0
take_packet_size_high5bits = 0x1f #本数据包大小 

arr11 = [46,48,35,30,46,48,35,30,46,48,35,30,32,59,52,65,96,83,95,89,85,87,80,81,63,69,67,74,45,43,38,20,19,14,9,7,2,10,12,5,16,18,23,55,92,76,78,70,92,76,78,70,92,76,78,70]
arr11=[temzz-1 for temzz in arr11]
arr21 = [42,47,36,34,42,47,36,34,42,47,36,34,33,60,53,54,94,82,90,88,93,86,79,73,62,61,68,75,44,39,37,21,15,13,8,3,1,11,4,6,17,22,24,56,91,84,77,71,91,84,77,71,91,84,77,71]
arr21=[temzz2-1 for temzz2 in arr21]
print(len(arr11),len(arr21))
#以下文件为配置文件
send_file0 = r"D:\work\tpc_debug\上位机\software\test_20220219\1.dat"
send_file1 = r"D:\work\tpc_debug\上位机\software\test_20220219\2.dat"
send_file2 = r"D:\work\tpc_debug\上位机\software\test_20220219\3.dat"
send_flle3=[]
send_file4 = r"D:\work\tpc_debug\上位机\software\test_20220219\5.dat"
send_file5 = r"D:\work\tpc_debug\上位机\software\test_20220219\6.dat"
send_file6 = r"D:\work\tpc_debug\上位机\software\test_20220219\7.dat"
send_file7 = r"D:\work\tpc_debug\上位机\software\test_20220219\8.dat"
send_file8=[]
send_file9=[]
send_file10=[]
send_file11 = r"D:\work\tpc_debug\上位机\处理代码\multi_hit.dat"

send_config0=r"D:\work\tpc_debug\上位机\处理代码\屏蔽0705\1_16.dat"
send_config1=r"D:\work\tpc_debug\上位机\处理代码\屏蔽0705\17_32.dat"
send_config2=r"D:\work\tpc_debug\上位机\处理代码\屏蔽0705\33_48.dat"
send_config3=r"D:\work\tpc_debug\上位机\处理代码\屏蔽0705\49_64.dat"
send_config4=r"D:\work\tpc_debug\上位机\处理代码\屏蔽0705\65_80.dat"
send_config5=r"D:\work\tpc_debug\上位机\处理代码\屏蔽0705\81_96.dat"


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #TCP协议
sock.settimeout(20)
file_size=1024000*20#每次采集文件大小 *10代表10M大小 
timeout_flag=1 
selfmode=0
collectstop=1

def connect():  #连接，请先选择模式和配置文件
    def tcp_sendfile(file_name,sock): #用于发文件
        with open(file_name, 'rb') as f:
            send_data1= f.read()
            sock.send(send_data1)
            print('have sent '+file_name)
        time.sleep(1)
        
    def tcp_config(sock):
        global selfmode
        tcp_sendfile(send_file7,sock)
        #time.sleep(1)

        for i in range(0,3):
            send_filename=eval("send_file"+str(i))
            tcp_sendfile(send_filename,sock)
           # time.sleep(1)
        
        if selfmode==0:
            if send_file8!=[] and send_file8!='' :
               tcp_sendfile(send_file8,sock) 
               #time.sleep(1)
            send_filename=eval("send_file"+str(3))
            tcp_sendfile(send_filename,sock)
            #time.sleep(1)
            tcp_sendfile(send_file11,sock)
            #time.sleep(1)

            if send_file9!=[] and send_file9!='':
               for m in range(10):
                  tcp_sendfile(send_file9,sock) 
                  time.sleep(0.5)
            if send_file10!=[]:
               tcp_sendfile(send_file10,sock) 
               #time.sleep(2)
        else:
            send_filename=eval("send_file"+str(3))
            tcp_sendfile(send_filename,sock)
            #time.sleep(1)

                
        #tcp_sendfile(send_file4,sock)
        

            
    global timeout_flag
    global selfmode
    for i in range(0,1):  #读取文件数量
        if timeout_flag==1:
            try:
               sock.connect(('192.168.10.16', 4660))
               print("当前与服务器重新连接成功.....")
               gt.insert( 'end','connecting sucess!\n')
               gt.update() 
            except:
                print('超时！')
                gt.insert( 'end','time error!\n')
                gt.update() 
            RECV_BUF_SIZE = 65535*32
    
            bsize = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print("Buffer size [Before]: %d" % bsize)
    
            sock.setsockopt( socket.SOL_SOCKET,socket.SO_RCVBUF,RECV_BUF_SIZE)
            bsize = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            print("Buffer size [after]: %d" % bsize)
    
            tcp_config(sock)
            timeout_flag=0

        tcp_sendfile(send_file4,sock)    
        index=f'{i}'
        if selfmode==1:
            name='自触发'
        else:
            name='击中'
            for i in range(0,6):
                send_filename=eval("send_config"+str(i))
                tcp_sendfile(send_filename,sock)
            
        
        abw= datetime.strftime(datetime.now(),'%Y%m%d-%H%M%S')#
        Recvfile_name=r'D:\work\tpc_debug\上位机\software\test_20220219\0'+str(abw)+str(name)+r'.dat'
       # Recvfile_name=r'D:\work\tpc_debug\上位机\software\test_20220219\'+str(abw) + r'.dat'
        with open(Recvfile_name,'wb') as fp:
            tcp_sendfile(send_file5,sock)
            tcp_sendfile(send_file6,sock)
            data_size=0
            time0=time.time()
            print('receiving data:'+Recvfile_name)
            op=0
            while True:
                if data_size < file_size:
                    try:
                        buffer = sock.recv(1024) #阻塞接收1024字节
                        op+=1
                        print(op)
                    except socket.timeout:    #超时处理逻辑
                        print("recving time out")
                        tcp_sendfile(send_file7,sock)
                        sock.close()
                        timeout_flag=1
                        break
                else:
                    break
                fp.write(buffer)
                data_size=data_size+len(buffer)
            if timeout_flag==0:
                tcp_sendfile(send_file7,sock)
        if getattr(sock,'_closed')==False:
            sock.close()
        time1=time.time()
        print('耗时：',time1-time0)
        gt.insert('end', '采集完成!\n')
        gt.update()
        gt.insert('end', '耗时：')
        gt.insert('end', time1-time0)
        gt.insert('end', '\n')
        gt.update()
   
def  self_mode(): #自触发
    global selfmode
    global send_file3
    selfmode=1
    send_file3= r"D:\work\tpc_debug\上位机\software\test_20220219\4_self.dat"
    print('模式',send_file3)
    gt.insert('end', '选择完成!\n')
    gt.update()

def hits_mode(): #击中
    global selfmode
    global send_file3
    selfmode=0
    send_file3= r"D:\work\tpc_debug\上位机\software\test_20220219\4_hit.dat"
    print('模式',send_file3)
    gt.insert('end', '选择完成!\n')
    gt.update()

def configcommand():#配置
    
    config_=Toplevel()
    config_.title('config')
    config_.geometry('600x600')

    global send_file8
    global send_file9
    global send_file10
    def setconfig0():
        global sss0
        sss0=filedialog.askopenfilename(title='导入时间阈值', filetypes=[('数据文件', '*.dat')])
        print(sss0)
    def setconfig1():
        global sss1
        sss1=filedialog.askopenfilename(title='导入触发阈值', filetypes=[('数据文件', '*.dat')])
        print(sss1)
   # def setconfig2():
        #global sss2
        #sss2=filedialog.askopenfilename(title='导入延迟', filetypes=[('数据文件', '*.dat')]) 
        #print(sss2)
        
    def setconfig():
        print(1)
        
        #print(sss0,sss1,sss2)
        global send_file8
        global send_file9
        global send_file10
        if sss0!='1':
           send_file8=sss0
           print('触发阈值:',send_file8)
           gt.insert('end', '触发阈值:'+send_file8+'\n')
           gt.update()  
        if sss1!='1':
           send_file9=sss1
           print('通道压缩阈值:',send_file9)
        #if sss2!=1:
          # send_file10=sss2
          # print('延迟:',send_file10)
        gt.insert('end', '配置完成!\n')
        gt.update()                 
        config_.destroy()
   
     
    global sss0
    global sss1
    global sss2   
    sss0='1'
    sss1='1'
    sss2='1'
    setconfigm = Button(config_, text="选择触发阈值",font=('宋体', 12), command=setconfig0)
    setconfigm.pack()
    setconfigm = Button(config_, text="选择通道压缩阈值阈值",font=('宋体', 12), command=setconfig1)
    setconfigm.pack()
    #setconfigm = Button(config_, text="选择延迟",font=('宋体', 12), command=setconfig2)
    #setconfigm.pack()
    setconfigm = Button(config_, text="配置",font=('宋体', 12), command=setconfig)
    setconfigm.pack()

    config_.mainloop() 

def loadcommand(): #导入数据
    sss=filedialog.askopenfilename(title='导入数据文件', filetypes=[('数据文件', '*.dat')])
    #sss="E:/lab/毕业论文/数据/20220418095113380_bkg_370V_800V_30min_120fC_25Mhz_500ns.dat"
    print(sss)
    f=open(sss,"rb")
    global waveform_data
    #alldata = f.read()  #alldata是数据
    waveform_data = np.fromfile(f, dtype=np.uint8)
    gt.insert( 'end','loading sucess!\n')
    gt.update()    
    gt.insert('end','Length:')
    gt.insert('end',len(waveform_data))
    gt.insert('end','\n')
    gt.update()  
    f.close()    

def unpack(): #解包
   '''
   每一个事例会分为三种格式的包：一个event header，多个（与这个事例读出的有效通道数相同）event body 和一个event end
   '''
   global waveform_data #数据
   global temp_data #空数组


   packet_header = np.where(waveform_data == take_packet_header)[0] #寻找头部对应的索引
   event_header_num = 0
   event_header = np.zeros(0)
   event_end_num = 0
   event_end = np.zeros(0)
   event_body_num = 0
   event_body = np.zeros(0)
   other_num = 0
   current_packet_length = 0
   
   # first header
   for i in range(len(packet_header)):
        if waveform_data[packet_header[i] + 1] != take_SOE_bit:#用于确认这个索引对应的是头部
            continue
        else:
            event_header_num += 1 #包头计数+1
            event_header = np.append(event_header, packet_header[i]) #存储为真正的头部索引
            #print(waveform_data[packet_header[i] + 2])
            current_packet_length = ((waveform_data[packet_header[i] + 1] & take_packet_size_high5bits) << 8 )+ waveform_data[packet_header[i] + 2]
            #记录包的长度
            valid_header = packet_header[i]  
            break
        
   # sort header and find eventbody #分类头部文件并寻找包体
   for i in range(1,len(packet_header)): 
       #包头
        if waveform_data[packet_header[i] + 1] == take_SOE_bit and (packet_header[i] - valid_header) == current_packet_length:
            event_header_num += 1 
            event_header = np.append(event_header, packet_header[i])
            current_packet_length = ((waveform_data[packet_header[i] + 1] & take_packet_size_high5bits) << 8 )+ waveform_data[packet_header[i] + 2]
            valid_header = packet_header[i]
        #包尾
        elif waveform_data[packet_header[i] + 1] == take_EOE_bit and (packet_header[i] - valid_header) == current_packet_length:
            event_end_num += 1
            event_end = np.append(event_end, packet_header[i])
            current_packet_length = ((waveform_data[packet_header[i] + 1] & take_packet_size_high5bits) << 8 )+ waveform_data[packet_header[i] + 2]
            valid_header = packet_header[i]
        #包体
        elif (waveform_data[packet_header[i] + 1] & take_high3bits) >> 5 == 0 and (packet_header[i] - valid_header) == current_packet_length:
            event_body_num += 1
            event_body = np.append(event_body, packet_header[i])
            current_packet_length = ((waveform_data[packet_header[i] + 1] & take_packet_size_high5bits) << 8) + waveform_data[packet_header[i] + 2]
            valid_header = packet_header[i]
        else:
            continue

   #event_id and timestamp
   timestamp =np.zeros(0)
   event_id  =np.zeros(0)
   for k in range(event_header_num):
            #print(waveform_data[int(event_header[k]) + 4]<<40,(waveform_data[int(event_header[k]) + 5]<<32),(waveform_data[int(event_header[k])+ 6]<<24),(waveform_data[int(event_header[k])+ 7]<<16),(waveform_data[int(event_header[k])+ 8]<<8))
            #print(waveform_data[int(event_header[k]) + 9])
            a=((waveform_data[int(event_header[k]) + 4]*(2.0**40))+ (waveform_data[int(event_header[k]) + 5]*(2.0**32))
               + (waveform_data[int(event_header[k])+ 6]*(2.0**24)) + (waveform_data[int(event_header[k])+ 7]*(2.0**16))+ 
               (waveform_data[int(event_header[k])+ 8]*(2.0**8))+ waveform_data[int(event_header[k]) + 9])
            
            timestamp=np.append(timestamp,a)
            b=((waveform_data[int(event_header[k]) + 10]<<24)
                               + (waveform_data[int(event_header[k]) + 11]<<16) 
                               + (waveform_data[int(event_header[k]) + 12]<<8)
                               + waveform_data[int(event_header[k]) + 13]
              )
            event_id =np.append(event_id,b)

   # event_data
   temp_data = np.zeros((800000, 1028))
   target_num = -1
   for j in range(event_body_num - 1):
        if j%100 == 0:
            print('{:.1f}%'.format(j/event_body_num*100))
            
            gt.insert( 'end','{:.1f}%'.format(j/event_body_num*100))
            gt.insert('end','\n')
            gt.update()   
        event_header_before_loc = np.where(event_header < event_body[j])[0]
        real_event_header_loc = event_header_before_loc[-1]
        if True:
            target_num = target_num + 1
            temp_data[target_num, 0] = waveform_data[int(event_body[j]) + 3]
            temp_data[target_num, 1] = waveform_data[int(event_body[j])  + 4]
            #print(temp_data[target_num, 1])
            temp_data[target_num, 2] = timestamp[real_event_header_loc]
            temp_data[target_num, 3] = event_id[real_event_header_loc]
            temp_data[target_num, 4:1029] = (waveform_data[int(event_body[j])  + 6:int(event_body[j])  + 2053:2] - 16) * 256 + \
              waveform_data[int(event_body[j])  + 7:int(event_body[j])  + 2054:2]
              
              
   temp_data_zero = np.where(temp_data[:, 0] == 0)[0]
   temp_data_zero_first = temp_data_zero[0]
   temp_data = np.delete(temp_data, temp_data_zero, axis=0)
   print(len(temp_data))
   gt.insert( 'end','finish!')
   gt.insert('end','\n')
   gt.update()


   global temp_data_sort
   temp_data_sort = temp_data[temp_data[:, 3].argsort(kind='stable')]
   waveform_data = temp_data_sort[:, 4:1027].T
   print('通道数：',len(temp_data_sort))
   # 找出最大值及其位置
   global max_data_position
   global max_data
   max_data = np.min(waveform_data, axis=0)
   max_data_position = np.argmin(waveform_data, axis=0)


   # 计算基线均值
   mean_data_start = np.mean(waveform_data[:25, :], axis=0)
   mean_data_end = np.mean(waveform_data[:25, :], axis=0)
   mean_data = np.min([mean_data_start, mean_data_end], axis=0)
   # 计算峰值
   peak_data = (max_data - mean_data).T

   # 事例划分
   global event_change_position
   event_change_stamp = np.diff(np.concatenate(([0], temp_data_sort[:, 3], [0])))
   event_change_position = np.where(event_change_stamp > 0)[0]
   print(event_change_position[0])
   temp_data=[]
   gt.insert( 'end','finish!')
   gt.insert('end','\n')
   gt.update()

def savedata(): #保存解包数据
   global temp_data_sort
   test = pd.DataFrame(data=temp_data_sort)
   test.to_csv(r'./解包数据.csv')
   gt.insert( 'end','finish!')
   gt.insert('end','\n')
   gt.update()
    
def wavesave(): #波形打包处理，这里存储时没有区分x与y维
   global temp_data_sort
   global event_change_position
   global max_data_position
   global max_data
   
   global pp  #存1024个采样点的信息
   global pp1 #通道
   global bb #存最大值对应的采样点
   global rr  #存事例号
   global dd #最大值
   print('event number:',len(event_change_position))
   m=0
   pp=[]  #存1024个采样点的信息
   pp1=[] #存通道
   bb=[]  #村最大值对应的采样点
   rr=[]  #存事例号
   dd=[]  #最大值
   for i in range(1,len(event_change_position)):
       mm=[]
       mmm=[]
       bbb=[]
       rrr=[]
       ddd=[]
       while m< len(temp_data_sort):
          if m <event_change_position[i]:
              mm.append(temp_data_sort[m, 4:1027])
              mmm.append(temp_data_sort[m, 1])
              bbb.append(max_data_position[m])
              ddd.append(max_data[m])
              rrr.append(temp_data_sort[m, 3])
              m+=1
              #print(1)
                  
          else:
              pp.append(mm)
              pp1.append(mmm)
              bb.append(bbb)
              dd.append(ddd)
              rr.append(rrr)
              if i==len(event_change_position)-1: #如果是最后一个
                 while m< len(temp_data):
                     mm.append(temp_data_sort[m, 4:1027])
                     mmm.append(temp_data_sort[m, 1])
                     bbb.append(max_data_position[m])
                     ddd.append(max_data[m])
                     rrr.append(temp_data_sort[m, 3])
                     m+=1
                 pp.append(mm)
                 pp1.append(mmm)
                 bb.append(bbb)
                 dd.append(ddd)
                 rr.append(rrr)
                 break
              break
   m1=0
   m2=0
   m3=0
   m4=0
   m5=0
   m6=0
   qqq1=0
   shili=[]
   for i in range(len(pp1)):
       if  max(bb[i])>600:
           qqq1+=1
           shili.append(rr[i])
      # if len(pp1[i])>9 :
       if len(pp1[i])>6 and len(pp1[i])<10: #and min(dd[i])>100:
           m1+=1
           print('1',rr[i])
           print(i,pp1[i])
       if len(pp1[i])<2:
           m2+=1
       if len(pp1[i])==2:
           m3+=1
       if len(pp1[i])==3:
           m4+=1
       if len(pp1[i])==4:
           m5+=1         
       if len(pp1[i])==5:
           m6+=1  
   print('600以后:',qqq1)  
   #print(shili)            
   print('>5:',m1)
   print('<2:',m2)
   print('2:',m3)
   print('3:',m4)
   print('4:',m5)
   print('5:',m6)
   print(len(pp1))
   #with open(r'E:\lab\αβ例子GUI\0410\波形.pkl', 'wb') as file:
      # pickle.dump(pp, file)
   #with open(r'E:\lab\αβ例子GUI\0410\通道.pkl', 'wb') as file:
      # pickle.dump(pp1, file)
   gt.insert( 'end','finish!')
   gt.insert('end','\n')
   gt.update()



def select3(): #筛除饱和的
    global pp  #存1024个采样点的信息
    global pp1 #通道
    global bb #存最大值对应的采样点
    global rr  #存事例号
    global dd #最大值
    pp_new=[]
    pp1_new=[]
    bb_new=[]
    rr_new=[]
    dd_new=[]
    print('饱和事例筛除')
    print('筛选前的事例数',len(pp))
    for i in range(len(pp)):
        m=0
        for ii in range(len(pp[i])):
            a=bb[i][ii]
            if a>970:
                continue
            elif pp[i][ii][a+50]-dd[i][ii]<50:
                m+=1
                break             
        if m==0:
            pp_new.append(pp[i])
            pp1_new.append(pp1[i])
            bb_new.append(bb[i])
            rr_new.append(rr[i])
            dd_new.append(dd[i])
    pp=pp1=bb=rr=dd=[]
    pp= pp_new
    pp1=pp1_new
    bb=bb_new
    rr=rr_new
    dd=dd_new
    print('筛选后的事例数',len(pp))
    m1=0
    m2=0
    m3=0
    m4=0
    m5=0
    m6=0
    qqq1=0
    shili=[]
    for i in range(len(pp1)):
       if  max(bb[i])>600:
           qqq1+=1
           shili.append(rr[i])
       if len(pp1[i])>=6: #and min(dd[i])>100:
           m1+=1
           print('事例号',rr[i])
           print(i,pp1[i])
       if len(pp1[i])<2:
           m2+=1
       if len(pp1[i])==2:
           m3+=1
       if len(pp1[i])==3:
           m4+=1
       if len(pp1[i])==4:
           m5+=1         
       if len(pp1[i])==5:
           m6+=1  
    print('600以后:',qqq1)  
   #print(shili)            
    print('>5:',m1)
    print('<2:',m2)
    print('2:',m3)
    print('3:',m4)
    print('4:',m5)
    print('5:',m6)
    print(len(pp1))

def select4(): #筛选位于300-600的
    global pp  #存1024个采样点的信息
    global pp1 #通道
    global bb #存最大值对应的采样点
    global rr  #存事例号
    global dd #最大值
    pp_new=[]
    pp1_new=[]
    bb_new=[]
    rr_new=[]
    dd_new=[]
    print('饱和事例筛除')
    print('筛选前的事例数',len(pp))
    for i in range(len(pp)):
        pp_new2=[]
        pp1_new2=[]
        bb_new2=[]
        rr_new2=[]
        dd_new2=[]
        m=0
        for ii in range(len(pp[i])):
            a=bb[i][ii]
            if a>300 and a<600:
               pp_new2.append(pp[i][ii])
               pp1_new2.append(pp1[i][ii])
               bb_new2.append(bb[i][ii])
               rr_new2.append(rr[i][ii])
               dd_new2.append(dd[i][ii])                

           
        if pp_new2!=[]:
            pp_new.append(pp_new2)
            pp1_new.append(pp1_new2)
            bb_new.append(bb_new2)
            rr_new.append(rr_new2)
            dd_new.append(dd_new2)
    pp=pp1=bb=rr=dd=[]
    pp= pp_new
    pp1=pp1_new
    bb=bb_new
    rr=rr_new
    dd=dd_new
    
    print('筛选后的事例数',len(pp1))



def select(): #边缘反符合
    global pp  #存1024个采样点的信息
    global pp1 #通道
    global bb #存最大值对应的采样点
    global rr  #存事例号
    global dd #最大值
    pp_new=[]
    pp1_new=[]
    bb_new=[]
    rr_new=[]
    dd_new=[]
    arr3=[51,49,62,67,55,50,61,63,5,21,19,27,6,13,20,26]
    print('边缘反符合')
    print('筛选前的事例数',len(pp))
    for i in range(len(pp)):
        m=0
        for ii in range(len(arr3)):
            try:
               if arr3[ii] in pp1[i]:
                   m+=1
            except:
                print(pp1[i])
                break
        if m==0:
            pp_new.append(pp[i])
            pp1_new.append(pp1[i])
            bb_new.append(bb[i])
            rr_new.append(rr[i])
            dd_new.append(dd[i])
    pp=pp1=bb=rr=dd=[]
    pp= pp_new
    pp1=pp1_new
    bb=bb_new
    rr=rr_new
    dd=dd_new
    
    print('筛选后的事例数',len(pp))
            
    
def select2(): #筛除反符合
    global pp  #存1024个采样点的信息
    global pp1 #通道
    global bb #存最大值对应的采样点
    global rr  #存事例号
    global dd #最大值
    pp_new=[]
    pp1_new=[]
    bb_new=[]
    rr_new=[]
    dd_new=[]
    arr4=[56,57,70,66,39,31,72,71,46,47,69,68,48,40,25,33]
    print('筛选前的事例数',len(pp))
    for i in range(len(pp)):
        m=0
        for ii in range(len(arr4)):
            try:
               if arr4[ii] in pp1[i]:
                   m+=1
                   break
            except:
                print(pp1[i])
                
        if m==0:
            pp_new.append(pp[i])
            pp1_new.append(pp1[i])
            bb_new.append(bb[i])
            rr_new.append(rr[i])
            dd_new.append(dd[i])
    pp=pp1=bb=rr=dd=[]
    pp= pp_new
    pp1=pp1_new
    bb=bb_new
    rr=rr_new
    dd=dd_new
    print('筛除反符合通道击中')
    print('筛选后的事例数',len(pp))
    m1=0
    m2=0
    m3=0
    m4=0
    m5=0
    m6=0
    for i in range(len(pp1)):
       if len(pp1[i])>5: #and min(dd[i])>100:
           m1+=1
           print('1',rr[i])
           print(i,pp1[i])
       if len(pp1[i])<2:
           m2+=1
       if len(pp1[i])==2:
           m3+=1
       if len(pp1[i])==3:
           m4+=1
       if len(pp1[i])==4:
           m5+=1         
       if len(pp1[i])==5:
           m6+=1  
       
    print('>5:',m1)
    print('<2:',m2)
    print('2:',m3)
    print('3:',m4)
    print('4:',m5)
    print('5:',m6)
    print(len(pp1))
 
    
def plotting():
    config_=Toplevel()
    config_.title('config')
    config_.geometry('600x600')
    

    def energy_spectrum(): #能谱
        global pp
        global dd
        energy=[]
        for i in range(len(pp)):
            energy0=0
            for ii in range(len(pp[i])):
                ever= np.mean(pp[i][ii][0:50])
                energy0+= ever-dd[i][ii]
            energy.append(energy0)
        plt.rcParams['font.sans-serif'] = ['SimHei']    
        plt.figure(figsize=(10,4),dpi=150)
        plt.title("能谱")
        plt.hist(energy, bins = 60)
        plt.show() 
        
    def waveall(): #画总波形，但是只画一半的事例
        plt.figure(figsize=(10,4),dpi=300)
        for i in range(len(pp)//2):
            for ii in range(len(pp[i])):
                col = (np.random.random(), np.random.random(), np.random.random())
                plt.plot(pp[i][ii], c=col,linewidth = 0.2)
        plt.show()   
    
    def location(): #画入射位置
        #import mpl_scatter_density
        #from scipy.stats import gaussian_kde
        global bb #最大值的采样点
        global pp1 #通道
        x_loc=[]
        y_loc=[]
        for i in range(len(pp1)):
            x=[]
            x1=[]
            y=[]
            y1=[]
            for ii in range(len(pp1[i])):
                 a1=np.where(arr11 == pp1[i][ii])[0]
                 if a1.size > 0:
                     for m in a1:
                         x.append(m)
                         x1.append(bb[i][ii])
                 else:
                     a2=np.where(arr21 == pp1[i][ii])[0]
                     if a2.size > 0:
                         for m in a2:
                             y.append(m)
                             y1.append(bb[i][ii])
                     else:
                       # print('error!')
                        continue
            if x!=[] and y!=[]:
                a1=x1.index(max(x1))
                x0=x[a1]
                a2=y1.index(max(y1))
                y0=y[a2]
                x_loc.append(x0)
                y_loc.append(y0)
        print(len(x_loc))   
        print(x_loc)
        print(len(y_loc) )
        print(y_loc) 
        for i in range(0,56):
            print(i,y_loc.count(i))
        x_loc=np.array(x_loc)
        y_loc=np.array(y_loc)
        

        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig1 = plt.figure()
        plt.title("x维入射通道统计")
        plt.hist(x_loc,bins = 56,range=(0,56),rwidth=0.8,align='left')
        plt.show()
        fig2 = plt.figure()
        plt.title("y维入射通道统计")
        plt.hist(y_loc,bins = 56,range=(0,56),rwidth=0.8,align='left')
        plt.show()        
        counts,xedges,yedges=np.histogram2d(y_loc,x_loc,bins=30,range=[[0,56],[0,56]])
        print(counts)
        fig = plt.figure()
        plt.title('入射位置分布')
        plt.imshow(counts,extent=[0,56,0,56],origin='lower',cmap='viridis')
        #plt.scatter(x_loc,y_loc,c=counts.flatten(),s=5,cmap='viridis')
        plt.colorbar()
        #a=sns.kdeplot(x=x_loc, y=y_loc, fill=True, cmap="viridis", cbar=True,n_levels = 50,thresh=0.05)
       # a.patch.set_facecolor('black')
        plt.show()
        '''
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        ax.hexbin(x_loc, y_loc, gridsize=50, cmap='inferno')
        density = ax.scatter_density(x_loc, y_loc,filterrad=10, cmap='Spectral_r')
        #ax.set_xlim(0, 55)
        #ax.set_ylim(0, 55)
        fig.colorbar(density, label='Number of points per pixel')
        #  
        #plt.scatter(x_loc,y_loc,s = 5, alpha = 0.5, color = 'r')
        '''
        
    global pp  #存1024个采样点的信息
    global pp1 #通道
    global bb #存最大值对应的采样点
    global rr  #存事例号
    global dd #最大值
        
    setconfigm = Button(config_, text="总波形",font=('宋体', 12), command=waveall)
    setconfigm.pack()
    setconfigm = Button(config_, text="能谱",font=('宋体', 12), command=energy_spectrum)
    setconfigm.pack()
    setconfigm = Button(config_, text="入射位置",font=('宋体', 12), command=location)
    setconfigm.pack()
    config_.mainloop()                                 

                    
def channelcount(): #通道计数
    global temp_data_sort
    x=[]
    y=[]
    for i in range(len(temp_data_sort)):
      a=temp_data_sort[i,1]
      a1=np.where(arr11 == a)[0] #判断这个通道在x维还是y维
      #print(a1,a)
      if a1.size > 0:
          for m in a1:
              x.append(m)
      else:
          a2=np.where(arr21 == a)[0]
          if a2.size > 0:
              for m in a2:
                  y.append(m)
          else:
             #print('error!')
             continue
    m=0      
    for i in range(len(temp_data_sort)):
        if temp_data_sort[i,1]==11:
            m+=1
    for i in range(0,97):
            print(i,np.sum(temp_data_sort[:,1]==i))
    print('m:',m)
            
    plt.rcParams['font.sans-serif'] = ['SimHei']

    plt.figure(figsize=(10,4),dpi=150)
    plt.title("x维通道统计")
    plt.hist(x, bins = 56,range=(0,56),rwidth=0.8,align='left')
    plt.show() 
    plt.figure(figsize=(10,4),dpi=150)
    plt.title("y维通道统计")
    plt.hist(y, bins = 56,range=(0,56),rwidth=0.8,align='left')
    plt.show()     
    

def trackcommand():#用于径迹显示画图，画之前需要运行波形保存
    global bb #存最大值对应的采样点
    global pp1 #通道
    global pp #存1024个采样点的信息
    global dd ##最大值
    
    top=Toplevel() #弹出窗口
    top.title('wave') #窗口标题
    top.geometry('800x800') #窗口大小
    #top.resizable(False,False) #设置窗口的宽和高是否可以改变，不可改变
    graph=Canvas(top,width=800,height=600,background='black') #绘图
    #graph=Canvas(top,background='black') #绘图
    graph.pack() #expand=YES,fill=BOTH
    chosenevent=StringVar() #StringVar()是一种变量类型，可以跟踪变量的值的变化，以保证值的变更随时可以显示在界面上。
    eventlist_name = Label(top, text='事件：')
    eventlist=ttk.Combobox(top,textvariable=chosenevent)
    eventlist.pack()
    eventlist["value"]=list(range(len(bb)))
    eventlist.current(0)   
        
    def drawstartcommand1():#画图
        nevent=int(eventlist.get())
        temp_x=[]
        temp_y=[]
        x=[]
        y=[]
        sizex=[]
        sizey=[]
        print(pp1[nevent])
        for i in range(len(pp1[nevent])):
            a1=np.where(arr11 == pp1[nevent][i])[0] #判断这个通道在x维还是y维
            if a1.size > 0:
                for m in a1:
                    x.append(m)
                    temp_x.append(bb[nevent][i])
                    ave=np.mean([pp[nevent][i][j] for j in range(0,50)])
                    sizex.append((ave-dd[nevent][i])/10)
            else:
                a2=np.where(arr21 == pp1[nevent][i])[0]
                if a2.size > 0:
                   for m in a2:
                       y.append(m)
                       temp_y.append(bb[nevent][i])
                       ave=np.mean([pp[nevent][i][j] for j in range(0,50)])
                       sizey.append((ave-dd[nevent][i])/10)
                else:
                    print('error!')
                    continue
        print(x)
        print('y',y)  
        print(len(x))
        print(temp_x)
        print(temp_y)            
        try:
          sorted_list1 = sorted(zip(x, temp_x,sizex))
          sorted_list2 = sorted(zip(y, temp_y,sizey))    
          x, temp_x,sizex=zip(*sorted_list1)
          y, temp_y,sizey=zip(*sorted_list2)
          sizex=list(sizex)
          sizey=list(sizey)
          sizex=[sizex[i] for i in range(len(sizex))]
          sizey=[sizey[i] for i in range(len(sizey))]
        except Exception as r:
            print(r)
            print(x)
            print(y)
            print(temp_x)
            print(temp_y)
        
        plt.figure(figsize=(10,4),dpi=150)
        #plt.ylim(200,450)
        plt.title("x")
        plt.scatter(x, temp_x, c="red",s=sizex,alpha=0.5)
        plt.plot(x, temp_x, c="blue",linewidth = 0.5)
        #plt.plot(x, temp_x, c="blue",linewidth = 0.5,marker='o',markersize=sizex, markerfacecolor='red',markeredgecolor='red')
        plt.show() 
        plt.figure(figsize=(10,4),dpi=150)
        #plt.ylim(0,1000)
        plt.title("y")
        plt.scatter(y, temp_y, c="red",s=sizey,alpha=0.5)
        plt.plot(y, temp_y, c="blue",linewidth = 0.5)
        #plt.plot(y, temp_y, c="blue",linewidth = 0.5,marker='o',markersize=sizey, markerfacecolor='red',markeredgecolor='red')
        plt.show()     

    def drawstartcommand2():#画图
        nevent=int(eventlist.get())
        temp_x=[]
        temp_y=[]
        x=[]
        y=[]
        print(pp1[nevent])
        for i in range(len(pp1[nevent])):
            a1=np.where(arr11 == pp1[nevent][i])[0] #判断这个通道在x维还是y维
            if a1.size > 0:
                for m in a1:
                    #x.append(m)
                    temp_x.append(pp[nevent][i])
            else:
                a2=np.where(arr21 == pp1[nevent][i])[0]
                if a2.size > 0:
                   for m in a2:
                       #y.append(m)
                       temp_y.append(pp[nevent][i])
                else:
                    print('error!')
                    continue
 
        print(temp_x)
        print(temp_y)       
        
        x=[i for i in range(0,1023)]
        print(len(x))
        
        #print(len(temp_x[0]))
        plt.figure(figsize=(10,4),dpi=150)
        plt.title("x")
        if temp_x !=[]:
          for  i in range(len(temp_x)):
               plt.plot(x, temp_x[i], c="blue",linewidth = 0.5,marker='o',markersize=1, markerfacecolor='red',markeredgecolor='red')
        plt.show()
        
        plt.figure(figsize=(10,4),dpi=150)
        plt.title("y")
        if temp_y !=[]:
          for  i in range(len(temp_y)):
               plt.plot(x, temp_y[i], c="blue",linewidth = 0.5,marker='o',markersize=1, markerfacecolor='red',markeredgecolor='red')
        plt.show()     
 
    def drawstartcommand3():#画图
        nevent=int(eventlist.get())
        temp_x=[]
        temp_y=[]
        x=[]
        y=[]
        print(pp1[nevent])
        for i in range(len(pp1[nevent])):
            a1=np.where(arr11 == pp1[nevent][i])[0] #判断这个通道在x维还是y维
            if a1.size > 0:
                for m in a1:
                    x.append(m)
                    temp_x.append(pp[nevent][i])
            else:
                a2=np.where(arr21 == pp1[nevent][i])[0]
                if a2.size > 0:
                   for m in a2:
                       y.append(m)
                       temp_y.append(pp[nevent][i])
                else:
                    print('error!')
                    continue
 
        print(temp_x)
        print(temp_y)       
        
        x0=[i for i in range(0,1023)]
        print(len(x0))
        
        plt.figure()
        plt.axes(projection='3d')
        plt.title("x")
        if temp_x !=[]:
          for  i in range(len(temp_x)):
               plt.plot(x0, x[i]*np.ones(len(x0)),temp_x[i])
        plt.show()
        
        plt.figure()
        plt.axes(projection='3d')
        plt.title("y")
        if temp_y !=[]:
          for  i in range(len(temp_y)):
               plt.plot(x0,y[i]*np.ones(len(x0)), temp_y[i])
        plt.show()        


    drawstart = Button(top, text="画图", command=drawstartcommand1)
    drawstart.pack()
    drawstart2 = Button(top, text="画事例总波形图", command=drawstartcommand2)
    drawstart2.pack()
    drawstart3 = Button(top, text="画三维图", command=drawstartcommand3)
    drawstart3.pack()
    top.mainloop()     
     
def drawcommand():#用于波形显示画图，
    global temp_data_sort #调用all，all是解包后的数据
    top=Toplevel() #弹出窗口
    top.title('wave') #窗口标题
    top.geometry('800x800') #窗口大小
    #top.resizable(False,False) #设置窗口的宽和高是否可以改变，不可改变
    graph=Canvas(top,width=800,height=600,background='black') #绘图
    #graph=Canvas(top,background='black') #绘图
    graph.pack() #expand=YES,fill=BOTH
    chosenevent=StringVar() #StringVar()是一种变量类型，可以跟踪变量的值的变化，以保证值的变更随时可以显示在界面上。
    eventlist_name = Label(top, text='事件：')
    eventlist=ttk.Combobox(top,textvariable=chosenevent)
    eventlist.pack()
    eventlist["value"]=list(range(len(temp_data_sort)))
    eventlist.current(0)
        
    def drawstartcommand():#画图
        nevent=int(eventlist.get())
        temp=[]
        x=[]
       # for mm  in range(0,100):
            #print(temp_data[mm][0],temp_data[mm][1],temp_data[mm][2],temp_data[mm][3])
        for i in range(4,1027):
            
            x.append(i)
            temp.append(temp_data_sort[nevent, i])
        fig = plt.figure(figsize=(7,4),dpi=150)
        #fig.patch.set_facecolor('black')
        ax = fig.add_subplot(111)
        ax.patch.set_alpha(0.0)
        plt.plot(x, temp, c="blue",linewidth = 0.5,marker='o',markersize=1, markerfacecolor='red',markeredgecolor='red')
        #plt.xticks([]) ####
        #plt.yticks([]) ####
        #plt.axis('off')
        plt.show()       

        #print(temp)
        #graph.create_line(temp,fill="white",width=3) 
        #print('1')
        #top.update_idletasks()
        #top.update()
        #print('1')
        
 
   
    drawstart = Button(top, text="画图", command=drawstartcommand)
    drawstart.pack()
    top.mainloop()

def channelcount2(): #通道计数,不区分x，y
    global temp_data_sort
    x=[]
    for i in range(len(temp_data_sort)):
      a=temp_data_sort[i,1]
      x.append(a)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(figsize=(10,4),dpi=150)
    plt.title("x维通道统计")
    plt.hist(x, bins = 96,range=(1,97))
    plt.xticks(np.arange(0,98,1),fontsize=5)
    plt.show() 
   

    
'''
创建Ui窗口
放置各个按钮,从上往下放置
这里的root是创建窗口对象，不是地址

'''
def blankspace(): #占行用
    condition=StringVar() #用于创建窗口
    condition_label=Label(root,textvariable=condition,
                      font=('宋体', 12), width=30, height=1) #用于创建窗口
    condition_label.pack() #用于创建窗口
    
root = Tk()                     # 创建窗口对象的背景色
#root.state("zoomed")
root.title('tpc project')       #用于创建窗口
root.geometry('1000x750')        #用于创建窗口

'''创建文本框并加滚动条
'''
blankspace()
gt = Text(root, height=20,width=120) 
scroll =Scrollbar()
scroll.pack(side=RIGHT,fill=Y)
scroll.config(command=gt.yview)
gt.config(yscrollcommand=scroll.set)
gt.pack()

b1= Button(root, text="自触发模式", font=('宋体', 12),command=self_mode) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
b2 = Button(root, text="测击中模式", font=('宋体', 12),command=hits_mode) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
b1.pack()
b2.pack() 

config = Button(root, text="配  置" ,font=('宋体', 12),command=configcommand)
config.pack()

up = Button(root, text="连接", font=('宋体', 12),command=connect) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

load = Button(root, text="导入数据",font=('宋体', 12), command=loadcommand)
load.pack() #pack用于防置组件，只能在四个位置放置组件，默认是放在top位置


up = Button(root, text="解  包", font=('宋体', 12),command=unpack) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

up = Button(root, text="保存数据", font=('宋体', 12),command=savedata) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

up = Button(root, text="通道计数", font=('宋体', 12),command=channelcount) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

up = Button(root, text="画单个波形图", font=('宋体', 12),command=drawcommand) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()
up = Button(root, text="波形打包", font=('宋体', 12),command=wavesave) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

up = Button(root, text="筛除饱和事例", font=('宋体', 12),command=select3) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

up = Button(root, text="筛选区间", font=('宋体', 12),command=select4) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

up = Button(root, text="边缘反符合", font=('宋体', 12),command=select) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

up = Button(root, text="反符合通道筛除", font=('宋体', 12),command=select2) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()

up = Button(root, text="画径迹图", font=('宋体', 12),command=trackcommand) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()
#up = Button(root, text="画入射位置图", font=('宋体', 12),command=location) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
#up.pack()
#up = Button(root, text="能谱", font=('宋体', 12),command=energy_spectrum) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
#up.pack()
up = Button(root, text="画图", font=('宋体', 12),command=plotting) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()
up = Button(root, text="不区分计数", font=('宋体', 12),command=channelcount2) #Button是tkinter框架中的一种按钮，通过点击可以调用后端函数。
up.pack()
blankspace()
root.mainloop()                 # 进入消息循环

