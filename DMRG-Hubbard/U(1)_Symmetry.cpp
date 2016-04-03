//
//  U(1)_Symmetry.cpp
//  DMRG-Hubbard
//
//  Created by Andrew Shen on 16/3/30.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//

#include "U(1)_Symmetry.hpp"

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd sort_indexes(VectorXd &v)
{

    // initialize original index locations
    VectorXd idx;
    idx.resize(v.size());
    for (size_t i = 0; i != idx.size(); i++) idx(i) = i;
    
    // sort indexes based on comparing values in v
    sort(idx.data(), idx.data() + idx.size(),
         [&v](size_t i1, size_t i2) {return v(i1) < v(i2);});
    
    sort(v.data(), v.data() + v.size());
    
    return idx;
}

VectorXd expand_vector(const VectorXd &v)
{
    VectorXd expanded_v(v.sum());
    
    int k = 0;
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v(i); j++) {
            expanded_v(k) = i;
            k++;
        }
    }
    
    return expanded_v;
}

VectorXd ordered_kron_vector(VectorXd &n1, VectorXd &n2, VectorXd &n)
{
    int d_n1 = n1.sum();
    int d_n2 = n2.sum();

    VectorXd kron_vect(d_n1 * d_n2);
    
    VectorXd e_n1 = expand_vector(n1);
    VectorXd e_n2 = expand_vector(n2);
    
    int k = 0;
    for (int i = 0; i < d_n1; i++) {
        for (int j = 0; j < d_n2; j++) {
            kron_vect(k) = e_n1(i) + e_n2(j);
            k++;
        }
    }
    
    VectorXd vec_index = sort_indexes(kron_vect);
    
    n.resize(kron_vect(kron_vect.size() - 1) + 1);
    n = VectorXd::Zero(kron_vect(kron_vect.size() - 1) + 1);
    
    for (int i = 0; i < d_n1 * d_n2; i++) {
        n(kron_vect(i))++;
    }

    return vec_index;
    
}

void reorder_matrix(MatrixXd &m, VectorXd &vec_idx)
{
    int dm = (int)m.cols();
    int dv = (int)vec_idx.size();
    
    //cout << "REORDER - d of matrix: " << dm << " d of idx: " << dv << endl;
    
    assert( (dm == dv) && "The size of the matrix and the vectors do not match! ");
    
    MatrixXd tmat(dm, dm);
    
    for (int i = 0; i < dm; i++) {
        for (int j = 0; j< dm; j++) {
            tmat(i, j) = m(vec_idx(i), vec_idx(j));
        }
    }
    
    m = tmat;
}

MatrixXd matrix_direct_plus(MatrixXd &m1, MatrixXd &m2)
{
    // May have problem with (n * 0) matrices
    int row_m1 = (int)m1.rows();
    int row_m2 = (int)m2.rows();
    int col_m1 = (int)m1.cols();
    int col_m2 = (int)m2.cols();
    
    MatrixXd nmat = MatrixXd::Zero(row_m1 + row_m2, col_m1 + col_m2);
    
    nmat.block(0, 0, row_m1, col_m1) = m1;
    nmat.block(row_m1, col_m1, row_m2, col_m2) = m2;
    
    return nmat;
    
}

int block_begin(VectorXd &v, int n)
{
    // Return -1 if the given particle number n does not exist.
    int tint = 0;
    int n_max = (int)v.size();
    if (n > n_max) {
        return -1;
    }
    if (v(n) == 0) {
        return -1;
    }
    
    for (int i = 0; i < n; i++) {
        tint += v(i);
    }
    
    return tint;
}

int block_end(VectorXd &v, int n)
{
    // Return -1 if the given particle number n does not exist.
    int tint = 0;
    int n_max = (int)v.size();
    if (n > n_max) {
        return -1;
    }
    if (v(n) == 0) {
        return -1;
    }
    
    for (int i = 0; i < n + 1; i++) {
        tint += v(i);
    }
    
    return tint - 1;
}

