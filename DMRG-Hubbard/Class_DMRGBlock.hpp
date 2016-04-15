//
//  Class_DMRGBlock.hpp
//  DMRG-Hubbard
//
//  Created by Andrew Shen on 16/3/27.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//


#ifndef Class_DMRGBlock_hpp
#define Class_DMRGBlock_hpp

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

enum class BlockPosition {LEFT,RIGHT};

MatrixXd matrix_direct_plus(MatrixXd &m1, MatrixXd &m2);
void matrix_reorder(MatrixXd &m, vector<int> &vec_idx);
vector<int> sort_index(vector<int> &vec);
vector<int> sort_index_double(vector<double> &vec);

class OperatorBlock
{
public:
    vector<MatrixXd> block;
    vector<int> QuantumN;
    vector<size_t> block_size;

    OperatorBlock();
    OperatorBlock(int _size);
    
    OperatorBlock &resize(int n);
    
    size_t size() const
    {
        return QuantumN.size();
    }
    
    size_t total_d()
    {
        size_t d = 0;
        for (int i = 0; i < this -> size(); i++) {
            d += block[i].cols();
        }
        return d;
    }
    
    int begin(int idx);
    int end(int idx);
    
    void CheckConsistency();
    void PrintInformation();
    
    void RhoPurification(const OperatorBlock &rho);
    void ZeroPurification();
    void Update(MatrixXd &m, vector<int> &qn);
    int SearchQuantumN(int n) const;

    MatrixXd Operator_full();
    vector<int> QuantumN_full();
};

vector<int> QuantumN_kron(OperatorBlock &ob1, OperatorBlock &ob2);
vector<size_t> SqueezeQuantumN(vector<int> &qn);
int SearchQuantumN(const vector<int>& qn, int n);
int b_begin(const vector<size_t>& block_size, int idx);
int SearchBlock(const vector<size_t>& block_size, int idx);

class SuperBlock : public OperatorBlock
{
public:
    void CheckConsistency();
    void Update(MatrixXd &m, vector<int> &qn);
        
    void PrintInformation();
    MatrixXd Operator_full();
};


class WavefunctionBlock
{
public:
    int quantumN_sector;
    vector<MatrixXd> block;
    vector<int> QuantumN;
    
    WavefunctionBlock();
    WavefunctionBlock(int _size);
    
    int SearchQuantumN(int n);
    
    WavefunctionBlock& resize(int n);
    
    size_t size() const
    {
        return QuantumN.size();
    }
    
    void PrintInformation();

    double norm();
    WavefunctionBlock& normalize();
    
    WavefunctionBlock operator+(const WavefunctionBlock& rhs);
    WavefunctionBlock operator-(const WavefunctionBlock& rhs);
    WavefunctionBlock operator*(double n);
    WavefunctionBlock operator/(double n);
    
    WavefunctionBlock& operator+=(const WavefunctionBlock& rhs);
    WavefunctionBlock& operator-=(const WavefunctionBlock& rhs);
    WavefunctionBlock& operator*=(double n);
    WavefunctionBlock& operator/=(double n);
    
    void Truncation(OperatorBlock& U, BlockPosition pos);
};


class DMRGBlock
{
public:
    OperatorBlock H;
    OperatorBlock U;

    vector<SuperBlock> c_up;
    vector<SuperBlock> c_down;
    
    vector<int> idx;
    
    DMRGBlock()
    {
        resize(1);
    }
    
    DMRGBlock(int _size)
    {
        resize(_size);
    }
    
    DMRGBlock &resize(int n)
    {
        c_up.resize(n);
        c_down.resize(n);
        
        return *this;
    }
    
    size_t size()
    {
        return c_up.size();
    }

};


#endif /* Class_DMRGBlock_hpp */
