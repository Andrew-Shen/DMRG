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
enum class SortOrder {ASCENDING, DESCENDING};

template <typename Type> void PrintVector(Type &vec);

MatrixXd MatrixDirectPlus(MatrixXd &m1, MatrixXd &m2);
void MatrixReorder(MatrixXd &m, vector<int> &vec_idx);

template <typename Type> vector<int> SortIndex(Type &vec, SortOrder so);

class OperatorBlock
{
public:
    vector<MatrixXd> block;
    vector<int> QuantumN;
    // Only for square blocks
    vector<size_t> block_size;

    OperatorBlock();
    OperatorBlock(int _size);
    
    OperatorBlock &resize(int n);
    
    size_t size() const
    {
        return QuantumN.size();
    }
    
    void RhoPurification(const OperatorBlock &rho);
    void ZeroPurification();
    void UpdateQN(const vector<int> &qn);
    void UpdateBlock(const MatrixXd &m);
    int SearchQuantumN(int n) const;

    MatrixXd FullOperator();
    vector<int> FullQuantumN();
    
    // Only for square blocks
    size_t total_d()
    {
        size_t d = 0;
        for (int i = 0; i < size(); i++) {
            d += block_size[i];
        }
        return d;
    }
    int BlockFirstIdx(int idx);
    int BlockLastIdx(int idx);
    
    // For debug
    void CheckConsistency();
    void PrintInformation();
};

vector<int> KronQuantumN(OperatorBlock &ob1, OperatorBlock &ob2);
vector<size_t> SqueezeQuantumN(vector<int> &qn);
int SearchIndex(const vector<int>& qn, int n);
int BlockFirstIndex(const vector<size_t>& block_size, int idx);
int SearchBlockIndex(const vector<size_t>& block_size, int idx);

class SuperBlock : public OperatorBlock
{
public:
    void UpdateBlock(const MatrixXd &m);
    
    MatrixXd FullOperator();

    void CheckConsistency();
    void PrintInformation();
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
    
    void UpdateQN(const vector<int>& qn, int _size);

};


#endif /* Class_DMRGBlock_hpp */
