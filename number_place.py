import time
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

class Cell():
    def __init__(self, row, col, val=0):
        self.row = row
        self.col = col
        self.block = (row-1)//3*3 + (col-1)//3 + 1
        self.__val = val
        if self.__val == 0:
            self.__possibles = set(range(1,10))
        else:
            self.__possibles = set()
    
    @property
    def val(self):
        return self.__val
    
    @val.setter
    def val(self, value):
        self.__val = value
        self.__possibles = set()

    @property
    def possibles(self):
        return self.__possibles
    
    def remove_from_possibles(self,val):
        if val in self.__possibles:
            self.__possibles.remove(val)
        
    def check_unique(self,):
        if self.val==0 and len(self.possibles)==1:
            self.val = list(self.possibles)[0]

def update_states(df):
    filled = set( cell.val for cell in df.cell if cell.val!=0 )
    for cell in df.cell:
        if cell.val==0:
            [ cell.remove_from_possibles(f) for f in filled ]

def find_unique_possibles(df):
    unique_possibles = []
    for i,row in df.iterrows():
        possibles = row.cell.possibles
        union_other_possibles = set(sum([ list(other.possibles) for j,other in zip(df.index,df.cell) if j!=row.name ],[]))
        unique_possible = [ possible for possible in possibles if possible not in union_other_possibles ]
        if len(unique_possible)==1:
            unique_possibles.append({'index':row.name, 'unique':unique_possible[0]})
        elif len(unique_possible)>1:
            print('Error!!')
            return []
    return unique_possibles

def fill_uniques(df,fill_list):
    for fill in fill_list:
        df.loc[fill['index'],'cell'].val = fill['unique']

def find_pairing_possibles(df):
    unique_pairings = []
    for i,row in df.iterrows():
        possibles = row.cell.possibles
        if len(possibles)==2:
            same_others = [ j for j,other in zip(df.index,df.cell) if j!=row.name and possibles==other.possibles ]
            if len(same_others)>0:
                unique_pairings.append((set([row.name]+same_others), possibles))
    return unique_pairings

def exclusive_pairings(df,pairing_list):
    for pairing in pairing_list:
        for i in df.index:
            if i not in list(pairing[0]):
                for val in pairing[1]:
                    df.loc[i,'cell'].remove_from_possibles(val)


class Board():
    def __init__(self, board_init=[]):
        self.reset(board_init)
    
    def reset(self, board_init=None):
        if board_init is not None:
            self.__board_init = board_init

        self.__cell_df = pd.DataFrame(columns=['row','col','block','cell'])
        k = 0
        for i in range(1,10):
            for j in range(1,10):
                self.__cell_df.loc[k,:] = [i,j,(i-1)//3*3+(j-1)//3+1,Cell(i,j,0)]
                k+=1

        for elem in self.__board_init:
            k = (elem['row']-1)*9 + elem['col']-1
            self.__cell_df.loc[k,'cell'].val = elem['val']
        
    def count_unfilled(self):
        return sum([ cell.val==0 for cell in self.__cell_df['cell']  ])
        
    def list_possibles(self):
        return [ cell.possibles if len(cell.possibles)>0 else 0 for cell in self.__cell_df['cell'] ]
    
    def check_uniques(self):
        [ cell.check_unique() for cell in self.__cell_df['cell'] ]

    def check_all_cols(self):
        for col in range(1,10):
            self.check_col(col)
            
    def check_col(self,col):
        update_states(self.__cell_df[self.__cell_df.col==col])
        unique_possibles = find_unique_possibles(self.__cell_df[self.__cell_df.col==col])
        fill_uniques(self.__cell_df, unique_possibles)
        update_states(self.__cell_df[self.__cell_df.col==col])

        pairing_list = find_pairing_possibles(self.__cell_df[self.__cell_df.col==col])
        exclusive_pairings(self.__cell_df[self.__cell_df.col==col], pairing_list)

    def check_all_rows(self):
        for row in range(1,10):
            self.check_row(row)
    
    def check_row(self,row):
        update_states(self.__cell_df[self.__cell_df.row==row])
        unique_possibles = find_unique_possibles(self.__cell_df[self.__cell_df.row==row])
        fill_uniques(self.__cell_df, unique_possibles)
        update_states(self.__cell_df[self.__cell_df.row==row])

        pairing_list = find_pairing_possibles(self.__cell_df[self.__cell_df.row==row])
        exclusive_pairings(self.__cell_df[self.__cell_df.row==row], pairing_list)
    
    def check_all_blocks(self):
        for block in range(1,10):
            self.check_block(block)
    
    def check_block(self,block):
        update_states(self.__cell_df[self.__cell_df.block==block])        
        unique_possibles = find_unique_possibles(self.__cell_df[self.__cell_df.block==block])
        fill_uniques(self.__cell_df, unique_possibles)
        update_states(self.__cell_df[self.__cell_df.block==block])
    
        pairing_list = find_pairing_possibles(self.__cell_df[self.__cell_df.block==block])
        exclusive_pairings(self.__cell_df[self.__cell_df.block==block], pairing_list)

    def step(self):
        self.check_all_rows()
        self.check_all_cols()
        self.check_all_blocks()
        self.check_uniques()
        
    def solve(self,):
        t0 = time.time()
        step = 0
        while self.count_unfilled()>0:
            self.step()
            step+=1
        print(f'time to solve: {time.time()-t0:.2f} [s]')
        print(f'steps to solve: {step}')
        self.draw()

    def draw(self,init=False):
        if init:
            board_arr = np.zeros((9,9), dtype=int)
            for elem in self.__board_init:
                row = elem['row']
                col = elem['col']
                val = elem['val']
                board_arr[row-1,col-1] = val
        else:
            board_arr = np.zeros((9,9), dtype=int)
            for cell in self.__cell_df.loc[:,'cell']:
                row = cell.row
                col = cell.col
                val = cell.val
                board_arr[row-1,col-1] = val

        plt.figure(figsize=(6,6))
        ax = sns.heatmap(board_arr, 
                         cmap='tab10', vmin=0, vmax=9,
                         square=True, 
                         annot=True, annot_kws={"size": 24},
                         linewidths=.5, linecolor="White",
                         cbar=False,
                        )
        ax.axis('off')
        ax.axhline(y=3, color='w',linewidth=5)
        ax.axhline(y=6, color='w',linewidth=5)
        ax.axvline(x=3, color='w',linewidth=5)
        ax.axvline(x=6, color='w',linewidth=5)
        plt.tight_layout()
        plt.show()

    def draw_init(self):
        self.draw(init=True)

