import numpy as np

try:

    def Entropy(p_l):
        P_pi,P_ni,total= p_n_no(p_l)
        result=((-P_pi/total)*np.log2(P_pi/total))-((-P_ni/total)*np.log2(P_ni/total))
        return result

    def p_n_no(l):
        pi=l.count('A')
        ni=l.count('B')
        total=pi+ni
        return pi,ni,total
    def Expected_Entropy(k,ch_l,Total):
        res=0
        for i in range(0,k):
            pi, ni,total= p_n_no(ch_l[i])
            # print(total)
            res1=(ni+pi)/Total
            res2=res1*Entropy(ch_l[i])
            res+=res2
        return res

    def Information_gain(p_l,ch_l,k,Total):
        H=Entropy(p_l)
        EH=Expected_Entropy(k,ch_l,Total)
        IG= H-EH
        print(f"entropy is :{H}")
        print(f"expected entropy for this split is : {EH}")
        print(f"information gain for this split will be :{IG}")


    def main():
        parent_labels=['A','A','B','B','B','B']
        Total=len(parent_labels)

        child_labels=[['A','B'],['A','B','B','B']] ## im putting the child node as list of lists to get child nodes number easily.
        k=len(child_labels)
        Information_gain(parent_labels,child_labels,k,Total)

    if  __name__=="__main__":
        main()
except Exception as e:
    print(e)
