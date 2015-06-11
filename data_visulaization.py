import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt

import collections
import csv
import numpy
def read_res(data_dict):
    res = collections.defaultdict(lambda : collections.defaultdict(lambda : None))
    for line in data_dict:
        cls, scale,ap=line.split()
        res[cls][scale]=ap
    return res

def cast_name(name):
    if "reduit_singlebb" in name:
        return name.replace("reduit_singlebb","std")
    elif "reduit_allbb" in name:
        return name.replace("reduit_allbb","eye")
    elif "ground" in name:
        return name.replace("ground","gt")


def read_curve_data(analysis_folder, curve_name, file_name,classes,scale):
    curve=[]
    curve_data = read_res(open("/".join([analysis_folder, curve_name, file_name])))
    for cls in classes:
        print curve_data[cls][scale]
        curve.append(eval(curve_data[cls][scale]))
    return curve


def draw_table(analysis_folder,file_name,curve_folder,classes,scale):
    with open(analysis_folder+'it50/'+scale+'.csv', 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['scale='+scale,'mean']+classes)
        for curve_name in curve_folder:
            curve = read_curve_data(analysis_folder, curve_name, file_name, classes, scale)
            spamwriter.writerow([cast_name(curve_name)]+[str(numpy.mean(curve))[:5]]+[str(c)[:5] for c in curve])


def draw_2D(analysis_folder,file_name,curve_folder,classes,scale,colors,line_style):
    x = np.array(range(len(classes)))
    my_xticks = classes
    plt.xticks(x, my_xticks, rotation=-30)
    plt.xlabel("class name")
    plt.ylabel("average precision")
    title="scale:"+str(scale)
    c = iter(colors)
    c_temp = next(c)
    ls = iter(line_style)
    for curve_name in curve_folder:
        curve = read_curve_data(analysis_folder, curve_name, file_name, classes, scale)
        title+=",avg("+cast_name(curve_name)+")="+str(sum(curve)/len(curve))[:5]
        ls_temp = next(ls,None)
        if ls_temp == None:        
            ls=iter(line_style)
            ls_temp=next(ls)
            c_temp=next(c)
        #multi line
        plt.plot(x, curve, color=c_temp, label=cast_name(curve_name),linestyle=ls_temp)
    
    plt.title(title)
    plt.legend()
    plt.grid()
#     plt.show()
    plt.savefig(analysis_folder+'it10/'+scale)
    plt.clf()


if __name__=="__main__":

    analysis_folder = "/local/wangxin/Data/ferrari_data/analysis/"
    file_name = "res_v3.txt"
    #convergence
    # curve_folder = ["it100/reduit_allbb/","it90/reduit_allbb/",
    #                 "it80/reduit_allbb/", "it70/reduit_allbb/", 
    #                 "it60/reduit_allbb/", "it50/reduit_allbb/",
    #                 "it40/reduit_allbb/","it30/reduit_allbb/",
    #                 "it20/reduit_allbb/","it10/reduit_allbb/",
    #                 "it5/reduit_allbb/",]
    
    # curve_folder = ["it50/reduit_allbb/", "it60/reduit_allbb/"
    #                 "it40/reduit_allbb/", "it30/reduit_allbb/"]
    # curve_folder = ["it50/reduit_allbb/", "it40/reduit_allbb/",]
    #curve_folder = ["it50/reduit_singlebb","it50/reduit_allbb"]
    
    curve_folder = ["it50/reduit_singlebb/","it10/reduit_allbb/","it50/reduit_allbb/"]
    classes=['cat', 'dog', 'bicycle', 'motorbike', 'boat', 'aeroplane', 'horse', 'cow', 'sofa', 'diningtable']
    # scales=["100","90","80","70","60","50"]
    scales=["100","90","80","70","60","50"]   
    colors=['b','g','r','c','m','y','k']
    #line_style=['-','--','-.',':']
    line_style=['-']
    for scale in scales:
        draw_2D(analysis_folder,file_name,curve_folder,classes,scale,colors,line_style)
        draw_table(analysis_folder,file_name,curve_folder,classes,scale)
        
    #curve_folder = ["it10/reduit_singlebb/","it10/reduit_allbb/"]
    #classes=['cat', 'boat']
    # scales=["100","90","80","70","60","50"]
    #scales=["100","90","80","70","60","50"]   
    #colors=['b','g','r','c','m','y','k']
    ##line_style=['-','--','-.',':']
    #line_style=['-']
    #for scale in scales:
    #    draw_2D(analysis_folder,file_name,curve_folder,classes,scale,colors,line_style)
    #    draw_table(analysis_folder,file_name,curve_folder,classes,scale)