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
            self.__candidates = set(range(1,10))
        else:
            self.__candidates = set()
    
    @property
    def val(self):
        return self.__val
    
    @val.setter
    def val(self, value):
        self.__val = value
        self.__candidates = set()

    @property
    def candidates(self):
        return self.__candidates
    
    def remove_from_candidates(self,val):
        if val in self.__candidates:
            self.__candidates.remove(val)
        
    def fill_unique(self,):
        if self.val==0 and len(self.candidates)==1:
            self.val = list(self.candidates)[0]

           
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
        return sum([ cell.val==0 for cell in self.__cell_df['cell'] ])
        
    def list_candidates(self):
        return [ cell.candidates if len(cell.candidates)>0 else 0 for cell in self.__cell_df['cell'] ]
    
    def check_uniques(self):
        [ cell.fill_unique() for cell in self.__cell_df['cell'] ]

    def check_all_cols(self):
        for col in range(1,10):
            self.check_col(col)
            
    def check_col(self,col):
        update_cells(self.__cell_df[self.__cell_df.col==col])
        unique_candidates = find_unique_candidates(self.__cell_df[self.__cell_df.col==col])
        fill_uniques(self.__cell_df, unique_candidates)
        update_cells(self.__cell_df[self.__cell_df.col==col])

        pairing_list = find_pairing_candidates(self.__cell_df[self.__cell_df.col==col])
        exclusive_pairings(self.__cell_df[self.__cell_df.col==col], pairing_list)

    def check_all_rows(self):
        for row in range(1,10):
            self.check_row(row)
    
    def check_row(self,row):
        update_cells(self.__cell_df[self.__cell_df.row==row])
        unique_candidates = find_unique_candidates(self.__cell_df[self.__cell_df.row==row])
        fill_uniques(self.__cell_df, unique_candidates)
        update_cells(self.__cell_df[self.__cell_df.row==row])

        pairing_list = find_pairing_candidates(self.__cell_df[self.__cell_df.row==row])
        exclusive_pairings(self.__cell_df[self.__cell_df.row==row], pairing_list)
    
    def check_all_blocks(self):
        for block in range(1,10):
            self.check_block(block)
    
    def check_block(self,block):
        update_cells(self.__cell_df[self.__cell_df.block==block])        
        unique_candidates = find_unique_candidates(self.__cell_df[self.__cell_df.block==block])
        fill_uniques(self.__cell_df, unique_candidates)
        update_cells(self.__cell_df[self.__cell_df.block==block])
    
        pairing_list = find_pairing_candidates(self.__cell_df[self.__cell_df.block==block])
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


def update_cells(df):
    filled = set( cell.val for cell in df.cell if cell.val!=0 )
    for cell in df.cell:
        if cell.val==0:
            [ cell.remove_from_candidates(f) for f in filled ]

def find_unique_candidates(df):
    unique_candidates = []
    for i, row in df.iterrows():
        candidates = row.cell.candidates
        union_other_candidates = set(sum([ list(other.candidates) for j,other in zip(df.index,df.cell) if j!=row.name ],[]))
        unique_candidate = [ candidate for candidate in candidates if candidate not in union_other_candidates ]
        if len(unique_candidate)==1:
            unique_candidates.append({'index':row.name, 'unique':unique_candidate[0]})
        elif len(unique_candidate)>1:
            print('Error!!')
            return []
    return unique_candidates

def fill_uniques(df,fill_list):
    for fill in fill_list:
        df.loc[fill['index'],'cell'].val = fill['unique']

def find_pairing_candidates(df):
    unique_pairings = []
    for i,cell in zip(df.index, df.cell):
        candidates = cell.candidates
        if len(candidates)==2:
            same_others = [ j for j,other in zip(df.index,df.cell) if j!=i and candidates==other.candidates ]
            if len(same_others)>0:
                unique_pairings.append({'indices':set([i]+same_others), 'candidates':candidates})
    return unique_pairings

def exclusive_pairings(df,pairing_list):
    for pairing in pairing_list:
        for i in df.index:
            if i not in list(pairing['indices']):
                for val in pairing['candidates']:
                    df.loc[i,'cell'].remove_from_candidates(val)
