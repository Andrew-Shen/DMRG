//
//  Class_DMRGBlock.cpp
//  DMRG-Heisenberg
//
//  Created by Andrew Shen on 16/3/27.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//


#include "Class_DMRGBlock.hpp"

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <math.h>

using namespace Eigen;
using namespace std;

void print(const vector<int> &vec)
{
    for (const auto& i: vec)
        cout << i << ' ';
    cout << endl;
}


MatrixXd matrix_direct_plus(MatrixXd &m1, MatrixXd &m2)
{
    size_t row_m1 = m1.rows();
    size_t row_m2 = m2.rows();
    size_t col_m1 = m1.cols();
    size_t col_m2 = m2.cols();
    
    if (row_m1 == 0 || col_m1 == 0) {
        assert(row_m1 == col_m1 && "Matrix direct plus: A (n x 0) or (0 x n) matrix encountered! ");
    }
    if (row_m2 == 0 || col_m2 == 0) {
        assert(row_m2 == col_m2 && "Matrix direct plus: A (n x 0) or (0 x n) matrix encountered! ");
    }
    
    MatrixXd nmat = MatrixXd::Zero(row_m1 + row_m2, col_m1 + col_m2);
    
    nmat.block(0, 0, row_m1, col_m1) = m1;
    nmat.block(row_m1, col_m1, row_m2, col_m2) = m2;
    
    return nmat;
}

void matrix_reorder(MatrixXd &m, vector<int> &vec_idx)
{
    size_t dm = m.cols();
    size_t dv = vec_idx.size();
    
    assert(dm == m.rows() && "Matrix reorder: The matrix is not square! ");
    assert(dm == dv && "Matrix reorder: The size of the matrix and the index vector do not match! ");
    
    MatrixXd tmat(dm, dm);
    
    for (int i = 0; i < dm; i++) {
        for (int j = 0; j < dm; j++) {
            tmat(i, j) = m(vec_idx[i], vec_idx[j]);
        }
    }
    
    m = tmat;
}

vector<int> sort_index(vector<int> &vec)
{
    // Initialize original index locations
    vector<int> idx(vec.size());
    for (size_t i = 0; i != idx.size(); i++) idx[i] = i;
    
    // Sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&vec](size_t i1, size_t i2) {return vec[i1] < vec[i2];});
    
    // Sort v itself
    sort(vec.begin(), vec.end());
    
    return idx;
}

vector<int> QuantumN_kron(OperatorBlock &ob1, OperatorBlock &ob2);


OperatorBlock::OperatorBlock()
{
    resize(0);
}

OperatorBlock::OperatorBlock(int _size)
{
    resize(_size);
}

OperatorBlock &OperatorBlock::resize(int n)
{
    QuantumN.resize(n);
    block.resize(n);
    
    return *this;
}

void OperatorBlock::CheckConsistency()
{
    assert(QuantumN.size() == block.size() && "OperatorBlock: An inconsistency in the quantum number and the actual block. ");
    
    for (int i = 0; i < QuantumN.size(); i++) {
        assert(block.at(i).cols() == block.at(i).rows() && "OperatorBlock: A non-square block! ");
        if (block.at(i).cols() == 0) {
            cout << "OperatorBlock: A block with zero size encountered. OperatorBlock::ZeroPurification is recommended. " << endl;
        }
    }
}

void OperatorBlock::ZeroPurification()
{
    vector<int>::iterator it_qn = QuantumN.begin();
    
    for(vector<MatrixXd>::iterator it_block = block.begin(); it_block != block.end(); )
    {
        if(it_block -> size() == 0)
        {
            it_block = block.erase(it_block); // It is very important to return the iterator.
            it_qn = QuantumN.erase(it_qn);
        }
        else
        {
            it_block++;
            it_qn++;
        }
    }
}

MatrixXd OperatorBlock::Operator_full()
{
    //this -> ZeroPurification();
    
    MatrixXd tmat;
    
    for (int i = 0; i < this -> size(); i++) {
        tmat.noalias() = matrix_direct_plus(tmat, block.at(i));
    }
    
    return tmat;
}

vector<int> OperatorBlock::QuantumN_full()
{
    vector<int> expanded_qn;
    
    for (int i = 0; i < QuantumN.size(); i++) {
        for (int j = 0; j < block.at(i).cols(); j++) {
            expanded_qn.push_back(QuantumN.at(i));
        }
    }
    
    return expanded_qn;
}

vector<int> QuantumN_kron(OperatorBlock &ob1, OperatorBlock &ob2)
{
    vector<int> kron_qn;
    
    vector<int> qn1 = ob1.QuantumN_full();
    vector<int> qn2 = ob2.QuantumN_full();
    
    size_t d1 = qn1.size();
    size_t d2 = qn2.size();
    
    for (int i = 0; i < d1; i++) {
        for (int j = 0; j < d2; j++) {
            kron_qn.push_back(qn1.at(i) + qn2.at(j));
        }
    }
    
    return kron_qn;
}

void OperatorBlock::Update(MatrixXd &m, vector<int> &qn)
{
    size_t dqn = qn.size();
    
    assert(m.cols() == m.rows() && "Update OperatorBlock: A non-square matrix! ");
    assert((m.cols() == dqn) && "Update OperatorBlock: Dimensions of matrix and quantum number vector do not agree! ");
    
    QuantumN.clear();
    block.clear();
    
    int flag_qn = qn[0];
    size_t pos = 0;
    size_t block_size = 1;
    for (int i = 1; i < dqn; i++) {
        if (qn.at(i) != flag_qn) {
            QuantumN.push_back(flag_qn);
            block.push_back(m.block(pos, pos, block_size, block_size));
            pos += block_size;
            block_size = 1;
            flag_qn = qn.at(i);
            continue;
        }
        block_size++;
    }
    
    // Special treatment for the last quantum number
    QuantumN.push_back(flag_qn);
    block.push_back(m.block(pos, pos, block_size, block_size));
    
}






WavefunctionBlock::WavefunctionBlock()
{
    resize(0);
}

WavefunctionBlock::WavefunctionBlock(int _size)
{
    resize(_size);
}

WavefunctionBlock &WavefunctionBlock::resize(int n)
{
    total_particle_number = n;
    
    block.resize(n + 1);
    
    return *this;
}

double WavefunctionBlock::norm()
{
    double n = 0;
    for (int i = 0; i <= total_particle_number; i++) {
        if (block[i].size() == 0) {
            continue;
        }
        n += block[i].norm() * block[i].norm();
    }
    return sqrt(n);
}

WavefunctionBlock &WavefunctionBlock::normalize()
{
    double n = WavefunctionBlock::norm();
    
    for (int i = 0; i <= total_particle_number; i++) {
        block[i] = block[i] / n;
    }
    
    return *this;
}


WavefunctionBlock WavefunctionBlock::operator*(double n)
{
    //WavefunctionBlock* tpsi = new WavefunctionBlock(this -> total_particle_number);
    WavefunctionBlock tpsi(total_particle_number);
    
    for (int i = 0; i <= total_particle_number; i++) {
        tpsi.block[i] = n * block[i];
    }
    
    return tpsi;
}

WavefunctionBlock WavefunctionBlock::operator/(double n)
{
    WavefunctionBlock tpsi(total_particle_number);
    
    for (int i = 0; i <= total_particle_number; i++) {
        tpsi.block[i] = block[i] / n;
    }
    
    return tpsi;
}

WavefunctionBlock WavefunctionBlock::operator+(WavefunctionBlock x)
{
    //WavefunctionBlock* tpsi = new WavefunctionBlock(this -> total_particle_number);
    WavefunctionBlock tpsi(total_particle_number);
    
    assert(tpsi.total_particle_number == x.total_particle_number && "Particle number incosistent! ");
    
    for (int i = 0; i <= total_particle_number; i++) {
        assert(block[i].cols() == x.block[i].cols() && "Matrix incosistent! ");
        assert(block[i].rows() == x.block[i].rows() && "Matrix incosistent! ");
        tpsi.block[i] = block[i] + x.block[i];
    }
    
    return tpsi;
}

WavefunctionBlock WavefunctionBlock::operator-(WavefunctionBlock x)
{
    //WavefunctionBlock* tpsi = new WavefunctionBlock(this -> total_particle_number);
    WavefunctionBlock tpsi(total_particle_number);
    
    assert(tpsi.total_particle_number == x.total_particle_number && "Particle number incosistent! ");
    
    for (int i = 0; i <= total_particle_number; i++) {
        assert(block[i].cols() == x.block[i].cols() && "Matrix incosistent! ");
        assert(block[i].rows() == x.block[i].rows() && "Matrix incosistent! ");
        tpsi.block[i] = block[i] - x.block[i];
    }
    
    return tpsi;
}
