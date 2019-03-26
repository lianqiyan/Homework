from image_search import *
import wx
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os


class HelloFrame(wx.Frame):

    def __init__(self,*args,**kw):
        super(HelloFrame,self).__init__(*args,**kw)

        pnl = wx.Panel(self)

        self.pnl = pnl
        st = wx.StaticText(pnl, label="图像检索系统", pos=(150, 0))
        font = st.GetFont()
        font.PointSize += 10
        font = font.Bold()
        st.SetFont(font)

        # 选择图像文件按钮
        self.btn = wx.Button(pnl, -1, "选择图片")
        self.btn.Bind(wx.EVT_BUTTON, self.OnSelect)

        self.makeMenuBar()

        self.CreateStatusBar()
        self.SetStatusText("欢迎使用基于Tensorflow的CNN图像检索系统")

    def makeMenuBar(self):
        fileMenu = wx.Menu()
        helloItem = fileMenu.Append(-1, "&Hello...\tCtrl-H",
                                    "Help string shown in status bar for this menu item")
        fileMenu.AppendSeparator()

        exitItem = fileMenu.Append(wx.ID_EXIT)
        helpMenu = wx.Menu()
        aboutItem = helpMenu.Append(wx.ID_ABOUT)

        menuBar = wx.MenuBar()
        menuBar.Append(fileMenu, "&File")
        menuBar.Append(helpMenu, "Help")

        self.SetMenuBar(menuBar)

        self.Bind(wx.EVT_MENU, self.OnHello, helloItem)
        self.Bind(wx.EVT_MENU, self.OnExit, exitItem)
        self.Bind(wx.EVT_MENU, self.OnAbout, aboutItem)

    def OnExit(self, event):
        self.Close(True)

    def OnHello(self, event):
        wx.MessageBox("Hello again from wxPython")

    def OnAbout(self, event):
        """Display an About Dialog"""
        wx.MessageBox("This is a wxPython Hello World sample",
                      "About Hello World 2",
                      wx.OK | wx.ICON_INFORMATION)

    def OnSelect(self, event):
        # wildcard = "image source(*.jpg)|*.jpg|" \
        #            "Compile Python(*.pyc)|*.pyc|" \
        #            "All file(*.*)|*.*"
        # dialog = wx.FileDialog(None, "Choose a file", os.getcwd(),
        #                        "", wildcard, wx.ID_OPEN)
        # if dialog.ShowModal() == wx.ID_OK:
        if event.Id == self.btn.Id:
            classes = ['all_souls', 'ashmolean', 'balliol', 'bodleian', 'christ_church', 'cornmarket',
                       'hertford', 'keble', 'magdalen', 'pitt_rivers', 'radcliffe_camera']
            saved_path = r'F:\Paper reproduce\image_search\zip2\CNN.model-9950.meta'
            # gen_vec(saved_path, 1)
            vec_path = os.path.join(os.getcwd(), 'vec_256.h5')
            # print(cosine_dis(np.array([0,1]), np.array([1,0])))
            res = get_img_path(vec_path, saved_path, r'G:\Oxford\images', r'G:\Oxford\groundtruth', classes)

            st1 = wx.StaticText(self.pnl, label="输入图像", pos=(150, 80))
            st2 = wx.StaticText(self.pnl, label="结果图像1", pos=(550, 80))
            st3 = wx.StaticText(self.pnl, label="结果图像2", pos=(950, 80))
            st4 = wx.StaticText(self.pnl, label="结果图像3", pos=(1350, 80))
            font = wx.Font(18, wx.DECORATIVE, wx.NORMAL , wx.NORMAL)
            st1.SetFont(font)
            st2.SetFont(font)
            st3.SetFont(font)
            st4.SetFont(font)



            self.initimage1(name= res[3])
            self.initimage2(name=res[2])
            self.initimage3(name=res[1])
            self.initimage4(name=res[0])

    # 生成图片控件
    def initimage1(self, name):
        imageShow = wx.Image(name, wx.BITMAP_TYPE_ANY)
        sb = wx.StaticBitmap(self.pnl, -1, imageShow.ConvertToBitmap(), pos=(0,110), size=(400,400))
        return sb

    def initimage2(self, name):
        imageShow = wx.Image(name, wx.BITMAP_TYPE_ANY)
        sb = wx.StaticBitmap(self.pnl, -1, imageShow.ConvertToBitmap(), pos=(400,110), size=(400,400))
        return sb

    def initimage3(self, name):
        imageShow = wx.Image(name, wx.BITMAP_TYPE_ANY)
        sb = wx.StaticBitmap(self.pnl, -1, imageShow.ConvertToBitmap(), pos=(800, 110), size=(400, 400))
        return sb

    def initimage4(self, name):
        imageShow = wx.Image(name, wx.BITMAP_TYPE_ANY)
        sb = wx.StaticBitmap(self.pnl, -1, imageShow.ConvertToBitmap(), pos=(1200, 110), size=(400, 400))
        return sb


if __name__ == '__main__':

    app = wx.App()
    frm = HelloFrame(None, title='Image_Search', size=(1600,600))
    frm.Show()
    app.MainLoop()
