import time
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

class Cell():
    def __init__(self, row, col, id, val=0):
        self.row = row
        self.col = col
        self.id = id
        self.block = (row-1)//3*3 + (col-1)//3 + 1
        self.__val = val
        if self.__val==0:
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
        self.__solver = Solver()
        self.reset(board_init)
    
    def reset(self, board_init=None):
        if board_init is not None:
            self.__board_init = board_init

        self.__cell_df = pd.DataFrame(columns=['row','col','block','val'])
        k = 0
        for i in range(1,10):
            for j in range(1,10):
                self.__cell_df.loc[k,:] = [i,j,(i-1)//3*3+(j-1)//3+1,0]
                k+=1

        for elem in self.__board_init:
            k = (elem['row']-1)*9 + elem['col']-1
            self.__cell_df.loc[k,'val'] = elem['val']

        self.__solver.reset(self.__board_init)
    
    def count_unfilled(self):
        return sum([ val==0 for val in self.__cell_df.val ])
        
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
            for row, col, val in zip(self.__cell_df.row,self.__cell_df.col,self.__cell_df.val):
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

    def list_candidates(self):
        return [ cell.candidates if len(cell.candidates)>0 else 0 for cell in self.__cells ]

    def solve(self):
        self.__solver.solve()
        self.update_board()
        self.draw()

    def step_solve(self):
        self.__solver.step()
        self.update_board()

    def update_board(self):
        self.__cell_df.val = self.__solver.get_values()


class Solver():
    def __init__(self, board_init=[]):
        self.reset(board_init)

    def reset(self, board_init):
        self.__cells = []
        k = 0
        for i in range(1,10):
            for j in range(1,10):
                self.__cells.append(Cell(i,j,k,0))
                k+=1

        for elem in board_init:
            k = (elem['row']-1)*9 + elem['col']-1
            self.__cells[k].val = elem['val']

    def count_unfilled(self):
        return sum([ cell.val==0 for cell in self.__cells ])

    def get_values(self):
        return [ cell.val for cell in self.__cells ]
  
    def check_uniques(self):
        [ cell.fill_unique() for cell in self.__cells ]

    def check_all_cols(self):
        for col in range(1,10):
            selected_cells = [ cell for cell in self.__cells if cell.col==col ]
            self.check_nine(selected_cells)
            
    def check_all_rows(self):
        for row in range(1,10):
            selected_cells = [ cell for cell in self.__cells if cell.row==row ]
            self.check_nine(selected_cells)
    
    def check_all_blocks(self):
        for block in range(1,10):
            selected_cells = [ cell for cell in self.__cells if cell.block==block ]
            self.check_nine(selected_cells)

    def check_nine(self,selected_cells):
        update_candidates_in_nine(selected_cells)        
        unique_candidates = find_unique_candidates(selected_cells)
        self.fill_uniques(unique_candidates)
        update_candidates_in_nine(selected_cells)
        pairing_list = find_pairing_candidates(selected_cells)
        exclusive_pairings(selected_cells, pairing_list)
        update_candidates_in_nine(selected_cells)

    def fill_uniques(self,fill_list):
        for fill in fill_list:
            self.__cells[fill['index']].val = fill['unique']

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


def update_candidates_in_nine(cells):
    assert len(cells)==9
    filled = set( cell.val for cell in cells if cell.val!=0 )
    for cell in cells:
        if cell.val==0:
            [ cell.remove_from_candidates(f) for f in filled ]

def find_unique_candidates(cells):
    unique_candidates = []
    for cell in cells:
        candidates = cell.candidates
        union_other_candidates = set(sum([ list(other.candidates) for other in cells if other.id!=cell.id ],[]))
        unique_candidate = [ candidate for candidate in candidates if candidate not in union_other_candidates ]
        if len(unique_candidate)==1:
            unique_candidates.append({'index':cell.id, 'unique':unique_candidate[0]})
        elif len(unique_candidate)>1:
            print('Error!!')
            return []
    return unique_candidates

def find_pairing_candidates(cells):
    unique_pairings = []
    for cell in cells:
        candidates = cell.candidates
        if len(candidates)==2:
            same_others = [ other.id for other in cells if other.id!=cell.id and candidates==other.candidates ]
            if len(same_others)>0:
                unique_pairings.append({'indices':set([cell.id]+same_others), 'candidates':candidates})
    return unique_pairings

def exclusive_pairings(cells,pairing_list):
    for pairing in pairing_list:
        for cell in cells:
            if cell.id not in list(pairing['indices']):
                for val in pairing['candidates']:
                    cell.remove_from_candidates(val)
