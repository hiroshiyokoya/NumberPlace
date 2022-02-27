# NumberPlace
Develop NumberPlace application 


## Classes
### Cell
describe cell status

#### Variables
- row
- col
- id
- block
- val: getter, setter
- candidates: getter

#### Methods
- remove_from_candidates: 
- fill_unique: 

### Board
describe board status

#### Variables
- __board_init
- __solver
- __cell_df

#### Methods
- reset
- count_unfilled
- draw
- draw_init
- list_candidates
- solve
- step_solve
- update_board

### Solver
solve the board

#### Variables
- __cells
#### Methods
- reset
- count_unfilled
- get_current_cell_values
- check_uniques
- check_all_cols
- check_all_rows
- check_all_blocks
- __check_nine
- fill_uniques
- step
- solve

