//
//  U(1)_Symmetry.hpp
//  DMRG-Hubbard
//
//  Created by Andrew Shen on 16/3/30.
//  Copyright © 2016年 Andrew Shen. All rights reserved.
//

#ifndef U_1__Symmetry_hpp
#define U_1__Symmetry_hpp

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd sort_indexes(VectorXd &v);
VectorXd expand_vector(const VectorXd &v);
VectorXd ordered_kron_vector(VectorXd &n1, VectorXd &n2, VectorXd &n);
void reorder_matrix(MatrixXd &m, VectorXd &vec_idx);
MatrixXd matrix_direct_plus(MatrixXd &m1, MatrixXd &m2);
int block_begin(VectorXd &v, int n);
int block_end(VectorXd &v, int n);


#endif /* U_1__Symmetry_hpp */
