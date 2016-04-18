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

template <typename Type>
void PrintVector(Type &vec)
{
    for (const auto& i: vec)
        cout << i << ' ';
    cout << endl;
    
}
template void PrintVector(const vector<int> &vec);
template void PrintVector(const vector<double> &vec);
template void PrintVector(const vector<size_t> &vec);

MatrixXd MatrixDirectPlus(MatrixXd &m1, MatrixXd &m2)
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

void MatrixReorder(MatrixXd &m, vector<int> &vec_idx)
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

template <typename Type>
vector<int> SortIndex(Type &vec, SortOrder so)
{
    // Initialize original index locations
    vector<int> idx(vec.size());
    for (size_t i = 0; i != idx.size(); i++) idx[i] = i;
    
        if (so == SortOrder::ASCENDING) {
            // Sort indexes based on comparing values in v
            sort(idx.begin(), idx.end(),
                 [&vec](size_t i1, size_t i2) {return vec[i1] < vec[i2];});
            // Sort v itself
            sort(vec.begin(), vec.end());
        } else {
            // Using rbegin() and rend() is eqivalent to sorting -vec
            // But *ineqivalent* to sorting vec with std::greater()
            sort(idx.rbegin(), idx.rend(),
                 [&vec](size_t i1, size_t i2) {return vec[i1] < vec[i2];});
            sort(vec.rbegin(), vec.rend());
        }
    
    return idx;
}
// To avoid linker error
template vector<int> SortIndex<vector<int>>(vector<int> &vec, SortOrder so);
template vector<int> SortIndex<vector<double>>(vector<double> &vec, SortOrder so);


vector<int> KronQuantumN(OperatorBlock &ob1, OperatorBlock &ob2)
{
    vector<int> kron_qn;
    
    vector<int> qn1 = ob1.FullQuantumN();
    vector<int> qn2 = ob2.FullQuantumN();
    
    size_t d1 = qn1.size();
    size_t d2 = qn2.size();
    
    for (int i = 0; i < d1; i++) {
        for (int j = 0; j < d2; j++) {
            kron_qn.push_back(qn1.at(i) + qn2.at(j));
        }
    }
    
    return kron_qn;
}

vector<size_t> SqueezeQuantumN(vector<int> &qn)
{
    size_t dqn = qn.size();
    
    vector<int> nqn;
    vector<size_t> block_size;

    int flag_qn = qn[0];
    size_t block_s = 1;
    for (int i = 1; i < dqn; i++) {
        if (qn[i] != flag_qn) {
            nqn.push_back(flag_qn);
            block_size.push_back(block_s);
            block_s = 1;
            flag_qn = qn[i];
            continue;
        }
        block_s++;
    }
    
    // Special treatment for the last quantum number
    nqn.push_back(flag_qn);
    block_size.push_back(block_s);
    
    qn = nqn;
    return block_size;
}

int SearchIndex(const vector<int>& qn, int n)
{
    for (int i = 0; i < qn.size(); i++) {
        if (qn[i] == n) {
            return i;
        }
    }
    // if there is no such element
    return -1;
}

int BlockFirstIndex(const vector<size_t>& block_size, int idx)
{
    // this function returns the first index of a given block
    size_t n_blocks = block_size.size();
    assert(idx > -1 && idx < n_blocks && "OperatorBlock: Index overbound! ");
    
    int res = 0;
    // rewritten with while
    for (int i = 0; i < n_blocks; i++) {
        if (i == idx) {
            break;
        }
        res += block_size[i];
    }
    return res;
}

int SearchBlockIndex(const vector<size_t>& block_size, int idx)
{
    // given the index of the *sorted* big QN, this function returns the block it belongs
    size_t n_blocks = block_size.size();
    
    int flag = 0;
    for (int i = 0; i < n_blocks; i++) {
        if (idx >= flag && idx < flag + block_size[i]) {
            return i;
        }
        flag += block_size[i];
    }
    assert("Something must be wrong! ");
    return ((int)n_blocks - 1);
}

OperatorBlock::OperatorBlock()
{
    resize(1);
}

OperatorBlock::OperatorBlock(int _size)
{
    resize(_size);
}

OperatorBlock &OperatorBlock::resize(int n)
{
    QuantumN.resize(n);
    block_size.resize(n);
    block.resize(n);
    
    return *this;
}

int OperatorBlock::BlockFirstIdx(int idx)
{
    return BlockFirstIndex(block_size, idx);
}

int OperatorBlock::BlockLastIdx(int idx)
{
    size_t n_blocks = size();
    assert(idx > -1 && idx < n_blocks && "OperatorBlock: Index overbound! ");
    
    int res;
    if (idx == n_blocks - 1) {
        res = total_d() - 1;
    } else {
        res = BlockFirstIdx(idx + 1) - 1;
    }
    return res;
}

void OperatorBlock::RhoPurification(const OperatorBlock &rho)
{
    vector<MatrixXd>::iterator it_block = block.begin();
    vector<size_t>::iterator it_bs = block_size.begin();
    
    for(vector<int>::iterator it_qn = QuantumN.begin(); it_qn != QuantumN.end(); )
    {
        if(rho.SearchQuantumN(*it_qn) == -1)
        {
            it_block = block.erase(it_block); // It is very important to return the iterator.
            it_qn = QuantumN.erase(it_qn);
            it_bs = block_size.erase(it_bs);
        }
        else
        {
            it_block++;
            it_qn++;
            it_bs++;
        }
    }
    
    for (int i = 0; i < this -> size(); i++) {
        assert(QuantumN[i] == rho.QuantumN[i] && "OperatorBlock: Quantum numbers of operator and rho do not match! ");
    }
}

void OperatorBlock::ZeroPurification()
{
    vector<int>::iterator it_qn = QuantumN.begin();
    vector<size_t>::iterator it_bs = block_size.begin();
    
    for(vector<MatrixXd>::iterator it_block = block.begin(); it_block != block.end(); )
    {
        if(*it_bs == 0) // the last "fictitious" block in the SuperBlock will not be removed in this way
        {
            it_block = block.erase(it_block); // It is very important to return the iterator.
            it_qn = QuantumN.erase(it_qn);
            it_bs = block_size.erase(it_bs);
        }
        else
        {
            it_block++;
            it_qn++;
            it_bs++;
        }
    }
}

MatrixXd OperatorBlock::FullOperator()
{
    MatrixXd tmat;
    
    for (int i = 0; i < this -> size(); i++) {
        tmat.noalias() = MatrixDirectPlus(tmat, block.at(i));
    }
    
    return tmat;
}

vector<int> OperatorBlock::FullQuantumN()
{
    vector<int> expanded_qn;
    
    for (int i = 0; i < QuantumN.size(); i++) {
        for (int j = 0; j < block.at(i).cols(); j++) {
            expanded_qn.push_back(QuantumN.at(i));
        }
    }
    
    return expanded_qn;
}

void OperatorBlock::UpdateQN(const vector<int> &qn)
{
    vector<int> tqn = qn;
    block_size = SqueezeQuantumN(tqn);
    QuantumN = tqn;
}

void OperatorBlock::UpdateBlock(const MatrixXd &m)
{
    assert(m.cols() == m.rows() && "Update OperatorBlock: A non-square matrix! ");
    assert((m.cols() == total_d()) && "Update OperatorBlock: Dimensions of matrix and quantum number vector do not agree! ");
    
    block.clear();

    size_t pos = 0;
    for (int i = 0; i < size(); i++) {
        size_t block_s = block_size[i];
        block.push_back(m.block(pos, pos, block_s, block_s));
        pos += block_s;
    }
}

int OperatorBlock::SearchQuantumN(int n) const
{
    return SearchIndex(QuantumN, n);
}

void OperatorBlock::CheckConsistency()
{
    size_t dqn = QuantumN.size();
    assert(dqn == block.size() && "OperatorBlock: An inconsistency in the quantum number and the actual block. ");
    //assert(dqn == block_size.size() && "OperatorBlock: Quantum number and block size do not agree! ");
    
    /*
    for (int i = 0; i < dqn; i++) {
        if (block[i].cols() != block[i].rows()) {
            cout << "OperatorBlock: A non-square block! " << endl;
        }
        assert(block[i].cols() == block_size[i] && "OperatorBlock: An inconsistency in the matrix size. ");
        if (block[i].cols() == 0) {
            cout << "OperatorBlock: A block with zero size encountered. OperatorBlock::ZeroPurification is recommended. " << endl;
        }
    }
     */
}

void OperatorBlock::PrintInformation()
{
    CheckConsistency();
    cout << "=========================" << endl;
    cout << this -> size() << " blocks in the OperatorBlock. With quantum numbers: " << endl;
    PrintVector(QuantumN);
    cout << "Corresponding matrix size: " << endl;
    PrintVector(block_size);
    //cout << "Operator Blocks: " << endl;
    //for (int i = 0; i < size(); i++) {
    //    cout << "Block " << i << ", Quantum number: " << QuantumN[i] << endl;
    //    cout << block[i] << endl;
    //}
    cout << "=========================" << endl;
}


void SuperBlock::UpdateBlock(const MatrixXd &m)
{
    assert(m.cols() == m.rows() && "Update OperatorBlock: A non-square matrix! ");
    assert((m.cols() == total_d()) && "Update OperatorBlock: Dimensions of matrix and quantum number vector do not agree! ");

    block.clear();

    int flag_qn = QuantumN[0];
    size_t pos = 0;
    for (int i = 1; i < QuantumN.size(); i++) {
        if (QuantumN.at(i) == flag_qn + 1) {
            block.push_back(m.block(pos, pos + block_size.at(i - 1), block_size.at(i - 1), block_size.at(i)));
        } else {
            block.push_back(MatrixXd::Zero(0, 0));
        }
        flag_qn = QuantumN.at(i);
        pos += block_size.at(i - 1);
    }
    // The last quantum number does not have coupling
    block.push_back(MatrixXd::Zero(0, 0));
}

MatrixXd SuperBlock::FullOperator()
{
    size_t total_d = 0;
    
    for (int i = 0; i < block_size.size(); i++) {
        total_d += block_size[i];
    }
    
    MatrixXd tmat = MatrixXd::Zero(total_d, total_d);
    
    int pos = 0;
    for (int i = 0; i < block_size.size(); i++) {
        tmat.block(pos, pos + block_size[i], block[i].rows(), block[i].cols()) = block[i];
        pos += block_size[i];
    }
    
    return tmat;
}

void SuperBlock::CheckConsistency()
{
    size_t dqn = QuantumN.size();
    int flag_qn = QuantumN[0];
    assert(dqn == block_size.size() && "SuperBlock: Quantum number and block size do not agree! ");
    for (int i = 1; i < dqn; i++) {
        if (QuantumN.at(i) == flag_qn + 1) {
            assert(block.at(i - 1).rows() == block_size.at(i - 1) && block.at(i - 1).cols() == block_size.at(i) && "SuperBlock: Matrix size does not match! ");
        } else {
            assert(block.at(i - 1).cols() == 0 && block.at(i - 1).rows() == 0 && "SuperBlock: A non-coupling quantum number has non-zero coupling matrix! ");
        }
        flag_qn = QuantumN.at(i);
    }
    
}

void SuperBlock::PrintInformation()
{
    CheckConsistency();
    cout << "=========================" << endl;
    cout << this -> size() << " blocks in the SuperBlock. With quantum numbers: " << endl;
    PrintVector(QuantumN);
    cout << "Corresponding matrix size: " << endl;
    PrintVector(block_size);
    cout << "Operator Blocks: " << endl;
    for (int i = 0; i < this -> size(); i++) {
        cout << "Block " << i << ", Quantum number: " << QuantumN[i] << endl;
        cout << block[i] << endl;
    }
}

WavefunctionBlock::WavefunctionBlock()
{
    quantumN_sector = 0;
    resize(0);
}

WavefunctionBlock::WavefunctionBlock(int _size)
{
    quantumN_sector = _size;
    resize(_size);
}

WavefunctionBlock &WavefunctionBlock::resize(int n)
{
    block.resize(n);
    QuantumN.resize(n);
    
    return *this;
}

void WavefunctionBlock::PrintInformation()
{
    cout << "=========================" << endl;
    cout << "Total quantum number of the WavefunctionBlock: " << quantumN_sector << endl;
    cout << size() << " blocks, with total quantum number: " << endl;
    PrintVector(QuantumN);
    cout << "Norm of this WavefunctionBlock: " << this -> norm() << endl;
    //cout << "Wavefunction Blocks: " << endl;
    //for (int i = 0; i < size(); i++) {
    //    cout << "Block " << i << ", Quantum number: " << QuantumN[i] << endl;
    //    cout << block[i] << endl;
    //}
}

double WavefunctionBlock::norm()
{
    double n = 0;
    for (int i = 0; i < this -> size(); i++) {
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
    
    for (int i = 0; i < this -> size(); i++) {
        block[i] = block[i] / n;
    }
    
    return *this;
}

int WavefunctionBlock::SearchQuantumN(int n)
{
    return SearchIndex(QuantumN, n);
}

WavefunctionBlock WavefunctionBlock::operator+(const WavefunctionBlock& rhs)
{
    assert(rhs.quantumN_sector == this -> quantumN_sector && "WavefunctionBlock Addition: Quantum number sector does not match! ");
    assert(rhs.size() == this -> size() && "WavefunctionBlock Subtraction: Quantum number sector does not match! ");
    
    WavefunctionBlock rval = WavefunctionBlock(*this);
    
    for (int i = 0; i < this -> size(); i++) {
        rval.block[i] += rhs.block[i];
    }
    
    return rval;
}

WavefunctionBlock WavefunctionBlock::operator-(const WavefunctionBlock& rhs)
{
    assert(rhs.quantumN_sector == this -> quantumN_sector && "WavefunctionBlock Addition: Quantum number sector does not match! ");
    assert(rhs.size() == this -> size() && "WavefunctionBlock Subtraction: Quantum number sector does not match! ");
    
    WavefunctionBlock rval = WavefunctionBlock(*this);
    
    for (int i = 0; i < this -> size(); i++) {
        rval.block[i] -= rhs.block[i];
    }
    
    return rval;
}

WavefunctionBlock WavefunctionBlock::operator*(double n)
{
    WavefunctionBlock rval = WavefunctionBlock(*this);
    
    for (int i = 0; i < this -> size(); i++) {
        rval.block[i] *= n;
    }
    
    return rval;
}

WavefunctionBlock WavefunctionBlock::operator/(double n)
{
    WavefunctionBlock rval = WavefunctionBlock(*this);
    
    for (int i = 0; i < this -> size(); i++) {
        rval.block[i] /= n;
    }
    
    return rval;
}

WavefunctionBlock& WavefunctionBlock::operator+=(const WavefunctionBlock& rhs)
{
    assert(rhs.quantumN_sector == this -> quantumN_sector && "WavefunctionBlock Addition: Quantum number sector does not match! ");
    assert(rhs.size() == this -> size() && "WavefunctionBlock Addition: Quantum number sector does not match! ");
    
    for (int i = 0; i < this -> size(); i++) {
        block[i] += rhs.block[i];
    }
    return *this;
}

WavefunctionBlock& WavefunctionBlock::operator-=(const WavefunctionBlock& rhs)
{
    assert(rhs.quantumN_sector == this -> quantumN_sector && "WavefunctionBlock Addition: Quantum number sector does not match! ");
    assert(rhs.size() == this -> size() && "WavefunctionBlock Subtraction: Quantum number sector does not match! ");
    
    for (int i = 0; i < this -> size(); i++) {
        block[i] -= rhs.block[i];
    }
    
    return *this;
}

WavefunctionBlock& WavefunctionBlock::operator*=(double n)
{
    for (int i = 0; i < this -> size(); i++) {
        block[i] *= n;
    }
    return *this;
}

WavefunctionBlock& WavefunctionBlock::operator/=(double n)
{
    for (int i = 0; i < this -> size(); i++) {
        block[i] /= n;
    }
    return *this;
}

void WavefunctionBlock::Truncate(OperatorBlock& U, BlockPosition pos, bool transposed)
{
    MatrixXd tmat;
    int U_idx;
    for (int i = 0; i < size(); i++) {
        if (pos == BlockPosition::LEFT) {
            U_idx = U.SearchQuantumN(QuantumN[i]);
            assert(U_idx != -1 && "WavefunctionBlock: Corresponding quantum number not found! ");
            if (transposed == true) {
                tmat = U.block[U_idx].transpose() * block[i];
            } else {
                tmat = U.block[U_idx] * block[i];
            }
        } else {
            U_idx = U.SearchQuantumN(quantumN_sector - QuantumN[i]);
            assert(U_idx != -1 && "WavefunctionBlock: Corresponding quantum number not found! ");
            if (transposed == true) {
                tmat = block[i] * U.block[U_idx].transpose();
            } else {
                tmat = block[i] * U.block[U_idx];
            }
        }
        block[i] = tmat;
    }
    // Zero purification
    vector<int>::iterator it_qn = QuantumN.begin();
    
    for(vector<MatrixXd>::iterator it_block = block.begin(); it_block != block.end(); )
    {
        if(it_block -> cols() == 0 || it_block -> rows() == 0)
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

void DMRGBlock::UpdateQN(const vector<int>& qn, int _size)
{
    resize(_size + 1);
    H.UpdateQN(qn);
    c_up[_size].UpdateQN(qn);
    c_down[_size].UpdateQN(qn);
}