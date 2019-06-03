def find_s(): 
    #实例集合 
    #变量一：孩子出生日天气
    #变量二：孩子出生月份
    #变量三：母亲属相
    #变量四：母亲年纪
    #变量五：母亲生日月份
    #变量六：父亲生日月份
    #变量七：0表示生男孩 1表示生女孩
    x1 = ['sunny', '1', '猪', '24', '6', '3' ,1] 
    x2 = ['sunny', '2', '猪', '29', '7', '2' , 1] 
    x3 = ['rainy', '3', '龙', '27', '8', '1', 0] 
    x4 = ['sunny', '4', '猪', '32', '9', '12', 1] 
    x5 = ['sunny', '5', '猪', '25', '10', '11', 1] 

    #训练样本集 
    xa = [x2,x3,x4,x1,x5] 
    h = [ None, None, None, None, None, None ] 
    for x in xa: 
        if x[6] == 1: 
            index = 0 
            for i in x[:-1]:
                if h[index] == None:
                    h[index] = i 
                elif h[index] != i: 
                     h[index] = '?'
                     
                index += 1

    print('规则是：',h) 
    
if __name__ == '__main__': find_s()
