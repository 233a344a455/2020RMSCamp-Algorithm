import read_picture  # line:11
import os  # line:1
import sys  # line:2
import time  # line:3
import random  # line:4
import _thread  # line:5
import tkinter as tk  # line:6
import tkinter.font as tf  # line:7
import tkinter.scrolledtext  # line:8
from link_search import link_search  # line:9
sys.path.append("../read_picture/")  # line:10


class auto_grader:  # line:14
    def __init__(O0O0O000O0O0O00OO, enable_ui=True):  # line:15
        if enable_ui:  # line:16
            _thread.start_new_thread(O0O0O000O0O0O00OO.ui, tuple())  # line:17
            time.sleep(1)  # line:18
        OOO00O00O00OO000O = '../mnist_data/t10k-images.idx3-ubyte'  # line:19
        O0O00000OOOOOOOO0 = '../mnist_data/t10k-labels.idx1-ubyte'  # line:20
        OO0O00OOO0O0O00OO = O0O0O000O0O0O00OO.random_image(
            OOO00O00O00OO000O, O0O00000OOOOOOOO0)  # line:21
        O0O0O000O0O0O00OO.load_images()  # line:22
        O0O0O000O0O0O00OO.ls = link_search(OO0O00OOO0O0O00OO)  # line:23
        O0O0O000O0O0O00OO.scores = [50, 20, 10, 0, -10, -100, -100]  # line:24
        O0O0O000O0O0O00OO.doublescores = [100, 20, 10, 0]  # line:25
        O0O0O000O0O0O00OO.__O0OO0OO0O00OOOO00 = 0  # line:27
        O0O0O000O0O0O00OO.round = 0  # line:28
        O0O0O000O0O0O00OO.enable_ui = enable_ui  # line:29

    def random_image(O0O0OO0OOOO0OO0OO, OOO0O0OO00OOO0000,
                     OOOOOOO0000000OO0):  # line:31
        OO0O0OOO0OOOO00OO, O0OOO0000000OO0OO = read_picture.read_image_data(
            OOO0O0OO00OOO0000, OOOOOOO0000000OO0)  # line:33
        OO0O0OO0O0O000000 = random.sample(range(0, len(OO0O0OOO0OOOO00OO)),
                                          32)  # line:34
        print('随机取样完成')  # line:35
        OO00O000O0O000OOO = dict()  # line:36
        for O00O00O0OOO0OOOO0 in OO0O0OO0O0O000000:  # line:37
            OO00O000O0O000OOO[O00O00O0OOO0OOOO0] = random.randint(0,
                                                                  2)  # line:38
        for OO0OOOO0OO0OO00O0 in range(len(OO0O0OO0O0O000000)):  # line:39
            O00O0O0OOO0OOO000 = O0OOO0000000OO0OO[
                OO0O0OO0O0O000000[OO0OOOO0OO0OO00O0]]  # line:40
            for O0OOO0OO00O000000 in range(
                    random.randint(0,
                                   len(OO0O0OOO0OOOO00OO) // 2),
                    len(OO0O0OOO0OOOO00OO)):  # line:43
                if O0OOO0000000OO0OO[
                        O0OOO0OO00O000000] == O00O0O0OOO0OOO000:  # line:44
                    OO0O0OO0O0O000000.append(O0OOO0OO00O000000)  # line:45
                    OO00O000O0O000OOO[O0OOO0OO00O000000] = OO00O000O0O000OOO[
                        OO0O0OO0O0O000000[OO0OOOO0OO0OO00O0]]  # line:46
                    break  # line:47
        random.shuffle(OO0O0OO0O0O000000)  # line:48
        OOO00O0OO0O000OOO = list()  # line:50
        for OO0OOOO0OO0OO00O0 in range(len(OO0O0OO0O0O000000)):  # line:51
            O00O00O0OOO0OOOO0 = OO0O0OO0O0O000000[OO0OOOO0OO0OO00O0]  # line:52
            OOO0O0OO0O00O00O0 = OO0O0OOO0OOOO00OO[O00O00O0OOO0OOOO0]  # line:53
            O00O0O0OOO0OOO000 = O0OOO0000000OO0OO[O00O00O0OOO0OOOO0]  # line:54
            OO00OOOOOO0000O0O = OO00O000O0O000OOO[O00O00O0OOO0OOOO0]  # line:55
            read_picture.image_save(OOO0O0OO0O00O00O0, OO00OOOOOO0000O0O,
                                    "../auto_grader/image/" +
                                    str(OO0OOOO0OO0OO00O0) + ".png")  # line:57
            OOO00O0OO0O000OOO.append([OO00OOOOOO0000O0O,
                                      O00O0O0OOO0OOO000])  # line:58
        print('图片显示完毕')  # line:59
        return OOO00O0OO0O000OOO  # line:60

    def index(O000O00OOO0O0O0O0, O0O000OO0O000O00O,
              OOOO00OO00O0OO000):  # line:62
        return O0O000OO0O000O00O * 8 + OOOO00OO00O0OO000  # line:63

    def ui(OOOO00OO0O00OOO00):  # line:65
        OOOO00OO0O00OOO00.blue = '#00BFFF'  # line:66
        OOOO00OO0O00OOO00.orange = 'orange'  # line:67
        OOOO00OO0O00OOO00.gray = '#3d3a3a'  # line:68
        OOOO00OO0O00OOO00.right_color = '#00FF00'  # line:69
        OOOO00OO0O00OOO00.wrong_color = 'red'  # line:70
        OOOO00OO0O00OOO00.unselected = 'white'  # line:71
        OOOO00OO0O00OOO00.selected = OOOO00OO0O00OOO00.orange  # line:72
        O00OOO0OOOOOOO0OO = tk.Tk()  # line:74
        O00OOO0OOOOOOO0OO.title(
            ' RoboMaster 2020 Summer Camp Algorithm AutoGrader')  # line:75
        # O00OOO0OOOOOOO0OO.iconbitmap('../auto_grader/favicon.ico')  #line:76
        O00OOO0OOOOOOO0OO.geometry('1000x650')  # line:77
        O00OOO0OOOOOOO0OO["background"] = 'white'  # line:78
        O00OOO0OOOOOOO0OO.resizable(0, 0)  # line:79
        OO0OOO00OOOOO0O00 = tk.Frame(O00OOO0OOOOOOO0OO,
                                     bg=OOOO00OO0O00OOO00.blue)  # line:81
        OO0OOO00OOOOO0O00.place(height=5, relwidth=0.25)  # line:82
        O0OO0O0OO0O00O00O = tk.Frame(O00OOO0OOOOOOO0OO,
                                     bg=OOOO00OO0O00OOO00.orange)  # line:83
        O0OO0O0OO0O00O00O.place(relx=0.25, height=5, relwidth=0.75)  # line:84
        OO0O00O0O00OOOOOO = tk.PhotoImage(
            file='../auto_grader/ROBOMASTER.png')  # line:86
        O00OO0000O000000O = tk.Label(O00OOO0OOOOOOO0OO,
                                     image=OO0O00O0O00OOOOOO,
                                     bg='white')  # line:87
        O00OO0000O000000O.place(x=28, y=25)  # line:88
        OOOOOO0OO0O0OOOOO = tk.Frame(O00OOO0OOOOOOO0OO, bg='white')  # line:90
        OOOOOO0OO0O0OOOOO.place(x=25, y=81, height=546, width=546)  # line:91
        OOOO00OO0O00OOO00.score_label = tk.Label(O00OOO0OOOOOOO0OO,
                                                 bg='white',
                                                 text='Score 0',
                                                 fg=OOOO00OO0O00OOO00.gray,
                                                 bd=0,
                                                 padx=0,
                                                 pady=0,
                                                 anchor='w',
                                                 font=('Microsoft YaHei UI',
                                                       32, tf.BOLD,
                                                       tf.ITALIC))  # line:102
        OOOO00OO0O00OOO00.score_label.place(x=626, y=15, width=306)  # line:103
        O0O000O0OO0O0000O = tk.scrolledtext.ScrolledText(O00OOO0OOOOOOO0OO,
                                                         width=70,
                                                         font=('微软雅黑',
                                                               14))  # line:107
        O0O000O0OO0O0000O.place(x=622, y=87, height=530, width=328)  # line:108
        sys.stdout = text_redirector(O0O000O0OO0O0000O, 'stdout')  # line:109
        OOOO00OO0O00OOO00.empty_photo = tk.PhotoImage(
            file='../auto_grader/empty.png').zoom(2)  # line:112
        OOOO00OO0O00OOO00.img_button_list = list()  # line:113
        OOOO00OO0O00OOO00.waitList = list()  # line:114
        for O0OO000O0OO000O0O in range(8):  # line:115
            for OOO0000O00OOOO0OO in range(8):  # line:116
                O0O0O0OO0OO0O00OO = tk.Button(
                    OOOOOO0OO0O0OOOOO,
                    bd=2,
                    bg='white',
                    state='disabled',
                    relief='flat',
                    image=OOOO00OO0O00OOO00.empty_photo,
                    command=(lambda O00O00O00OO0O0OO0, O00O00O00OOO0OO00:
                             lambda: OOOO00OO0O00OOO00.click_callback(
                                 O00O00O00OO0O0OO0, O00O00O00OOO0OO00))(
                                     O0OO000O0OO000O0O,
                                     OOO0000O00OOOO0OO))  # line:125
                O0O0O0OO0OO0O00OO.grid(row=O0OO000O0OO000O0O,
                                       column=OOO0000O00OOOO0OO,
                                       sticky=tk.W + tk.E + tk.N + tk.S,
                                       padx=3,
                                       pady=3)  # line:130
                OOOO00OO0O00OOO00.img_button_list.append(
                    O0O0O0OO0OO0O00OO)  # line:131
        O00OOO0OOOOOOO0OO.mainloop()  # line:132

    def load_images(OOO00000OOO00O0OO):  # line:134
        OOO00000OOO00O0OO.photoes = list()  # line:135
        for O0O000OOO00OOOOOO in range(64):  # line:136
            O0O00OOOO0O00OOO0 = tk.PhotoImage(file='../auto_grader/image/' +
                                              str(O0O000OOO00OOOOOO) +
                                              '.png').zoom(2)  # line:138
            OOO00000OOO00O0OO.photoes.append(O0O00OOOO0O00OOO0)  # line:139
            OOO00000OOO00O0OO.img_button_list[O0O000OOO00OOOOOO].config(
                image=O0O00OOOO0O00OOO0, state='normal')  # line:140

    def click_callback(OOO0OO00000O0OOOO, OO0O0O0O00O00OO0O,
                       O0O0O0O0O0O00O000):  # line:142
        print(OO0O0O0O00O00OO0O, O0O0O0O0O0O00O000)  # line:143
        O0O00OOOOOO000OOO = OOO0OO00000O0OOOO.index(
            OO0O0O0O00O00OO0O, O0O0O0O0O0O00O000)  # line:144
        if [OO0O0O0O00O00OO0O,
                O0O0O0O0O0O00O000] in OOO0OO00000O0OOOO.waitList:  # line:145
            OOO0OO00000O0OOOO.waitList.remove(
                [OO0O0O0O00O00OO0O, O0O0O0O0O0O00O000])  # line:146
            OOO0OO00000O0OOOO.img_button_list[O0O00OOOOOO000OOO].config(
                bg=OOO0OO00000O0OOOO.unselected)  # line:147
        else:  # line:148
            OOO0OO00000O0OOOO.waitList.append(
                [OO0O0O0O00O00OO0O, O0O0O0O0O0O00O000])  # line:149
            OOO0OO00000O0OOOO.img_button_list[O0O00OOOOOO000OOO].config(
                bg=OOO0OO00000O0OOOO.selected)  # line:150
            if (len(OOO0OO00000O0OOOO.waitList) == 2):  # line:151
                O0O00O000O00O00OO = OOO0OO00000O0OOOO.waitList[1]  # line:152
                OOO00O0O0000000OO = OOO0OO00000O0OOOO.waitList[0]  # line:153
                _thread.start_new_thread(
                    OOO0OO00000O0OOOO.link,
                    (O0O00O000O00O00OO[0], O0O00O000O00O00OO[1],
                     OOO00O0O0000000OO[0], OOO00O0O0000000OO[1]))  # line:155

    def color(OO00OOOO0OO00OOO0, OO00OO0OOOOOOO00O):  # line:157
        if OO00OO0OOOOOOO00O == 0:  # line:158
            return 'RED'  # line:159
        if OO00OO0OOOOOOO00O == 1:  # line:160
            return 'GREEN'  # line:161
        return 'BLUE'  # line:162

    def link(O000OO00O0O0O00OO, O0O00OOO0O000OO00, OOOOO00O000O0OOOO,
             OO0OO0OO0OO0OO0O0, O0OO0O00OO00OO0OO):  # line:164
        O000000O0OOOO0O00 = O000OO00O0O0O00OO.index(
            O0O00OOO0O000OO00, OOOOO00O000O0OOOO)  # line:165
        OO00O00OOO00O00OO = O000OO00O0O0O00OO.index(
            OO0OO0OO0OO0OO0O0, O0OO0O00OO00OO0OO)  # line:166
        if O000OO00O0O0O00OO.enable_ui:  # line:167
            time.sleep(0.5)  # line:168
        OOOOO00000O00O00O = O000OO00O0O0O00OO.ls.search(
            O0O00OOO0O000OO00, OOOOO00O000O0OOOO, OO0OO0OO0OO0OO0O0,
            O0OO0O00OO00OO0OO)  # line:169
        if isinstance(OOOOO00000O00O00O, list):  # line:170
            O000OO00O0O0O00OO.__O0OO0OO0O00OOOO00 += O000OO00O0O0O00OO.scores[
                5]  # line:171
            print('Wrong! {0}'.format(O000OO00O0O0O00OO.scores[5]))  # line:172
            print('{0}, {1} is {2} {3}'.format(
                O0O00OOO0O000OO00, OOOOO00O000O0OOOO,
                O000OO00O0O0O00OO.color(OOOOO00000O00O00O[0][0]),
                OOOOO00000O00O00O[0][1]))  # line:175
            print('{0}, {1} is {2} {3}'.format(
                OO0OO0OO0OO0OO0O0, O0OO0O00OO00OO0OO,
                O000OO00O0O0O00OO.color(OOOOO00000O00O00O[1][0]),
                OOOOO00000O00O00O[1][1]))  # line:178
            OOO000O0000OOOO0O = O000OO00O0O0O00OO.wrong_color  # line:179
        elif OOOOO00000O00O00O == -2:  # line:180
            O000OO00O0O0O00OO.__O0OO0OO0O00OOOO00 += O000OO00O0O0O00OO.scores[
                4]  # line:181
            print('Took more than four lines! {0}'.format(
                O000OO00O0O0O00OO.scores[4]))  # line:182
            OOO000O0000OOOO0O = O000OO00O0O0O00OO.wrong_color  # line:183
        elif OOOOO00000O00O00O == -1:  # line:184
            O000OO00O0O0O00OO.__O0OO0OO0O00OOOO00 += O000OO00O0O0O00OO.scores[
                6]  # line:185
            print('Invalid input! {0}'.format(
                O000OO00O0O0O00OO.scores[6]))  # line:186
            OOO000O0000OOOO0O = O000OO00O0O0O00OO.wrong_color  # line:187
        else:  # line:188
            O000OO00O0O0O00OO.round += 1  # line:189
            print('Round', O000OO00O0O0O00OO.round)  # line:190
            if O000OO00O0O0O00OO.round in [4, 8, 16, 27, 28, 29, 30, 31,
                                           32]:  # line:191
                O000OO00O0O0O00OO.__O0OO0OO0O00OOOO00 += O000OO00O0O0O00OO.doublescores[
                    OOOOO00000O00O00O - 1]  # line:192
                print('Right! +{0}'.format(
                    O000OO00O0O0O00OO.doublescores[OOOOO00000O00O00O -
                                                   1]))  # line:193
            else:  # line:194
                O000OO00O0O0O00OO.__O0OO0OO0O00OOOO00 += O000OO00O0O0O00OO.scores[
                    OOOOO00000O00O00O - 1]  # line:195
                print('Right! +{0}'.format(
                    O000OO00O0O0O00OO.scores[OOOOO00000O00O00O -
                                             1]))  # line:196
            OOO000O0000OOOO0O = O000OO00O0O0O00OO.right_color  # line:197
        print('Current score =',
              O000OO00O0O0O00OO.__O0OO0OO0O00OOOO00)  # line:198
        print('**********************')  # line:199
        if O000OO00O0O0O00OO.enable_ui:  # line:200
            O000OO00O0O0O00OO.score_label.config(text='Score {0}'.format(
                O000OO00O0O0O00OO.__O0OO0OO0O00OOOO00))  # line:201
            O000OO00O0O0O00OO.img_button_list[O000000O0OOOO0O00].config(
                bg=OOO000O0000OOOO0O)  # line:202
            O000OO00O0O0O00OO.img_button_list[OO00O00OOO00O00OO].config(
                bg=OOO000O0000OOOO0O)  # line:203
            time.sleep(0.75)  # line:204
            for O00O0O0OOOO000O0O in O000OO00O0O0O00OO.img_button_list:  # line:205
                O00O0O0OOOO000O0O.config(
                    bg=O000OO00O0O0O00OO.unselected)  # line:206
            O000OO00O0O0O00OO.waitList = []  # line:207
            if OOOOO00000O00O00O in [1, 2, 3, 4]:  # line:208
                O000OO00O0O0O00OO.img_button_list[O000000O0OOOO0O00].config(
                    image=O000OO00O0O0O00OO.empty_photo,
                    state='disabled')  # line:210
                O000OO00O0O0O00OO.img_button_list[OO00O00OOO00O00OO].config(
                    image=O000OO00O0O0O00OO.empty_photo,
                    state='disabled')  # line:212
        return OOOOO00000O00O00O  # line:213


class text_redirector():  # line:216
    def __init__(OO0O00O00000OO000, O00OO0000O00OOO0O,
                 OO0O0000O0O00O00O):  # line:217
        OO0O00O00000OO000.widget = O00OO0000O00OOO0O  # line:218
        OO0O00O00000OO000.tag = OO0O0000O0O00O00O  # line:219
        OO0O00O00000OO000.flush = sys.stdout.flush  # line:220

    def write(OO00O00OO00OOOOO0, OO0OOOOOOOOO000O0):  # line:222
        OO00O00OO00OOOOO0.widget.configure(state='normal')  # line:223
        OO00O00OO00OOOOO0.widget.insert('end', OO0OOOOOOOOO000O0,
                                        (OO00O00OO00OOOOO0.tag, ))  # line:224
        OO00O00OO00OOOOO0.widget.see(tk.END)  # line:225
        OO00O00OO00OOOOO0.widget.configure(state='disabled')  # line:226


if __name__ == "__main__":  # line:229
    ag = auto_grader()  # line:230
    import time
    time.sleep(1e8)
    # os.system('pause')  #line:231
