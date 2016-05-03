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

MatrixXcd MatrixDirectPlus(const MatrixXcd &m1, const MatrixXcd &m2);
void MatrixReorder(MatrixXcd &m, vector<int> &vec_idx);

template <typename Type> vector<int> SortIndex(Type &vec, SortOrder so);

class OperatorBlock
{
public:
    vector<MatrixXcd> block;
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
    void Truncate(const OperatorBlock &U);
    void ZeroPurification();
    void UpdateQN(const vector<int> &qn);
    void UpdateBlock(const MatrixXcd &m);
    int SearchQuantumN(int n) const;

    MatrixXcd FullOperator() const;
    vector<int> FullQuantumN() const;
    
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
    MatrixXcd IdentitySign();
    
    // For debug
    void CheckConsistency();
    void PrintInformation();
};

vector<int> KronQuantumN(const OperatorBlock &ob1, const OperatorBlock &ob2);
vector<size_t> SqueezeQuantumN(vector<int> &qn);
int SearchIndex(const vector<int>& qn, int n);
int BlockFirstIndex(const vector<size_t>& block_size, int idx);
int SearchBlockIndex(const vector<size_t>& block_size, int idx);

class SuperBlock : public OperatorBlock
{
public:
    void UpdateBlock(const MatrixXcd &m);
    void Truncate(const OperatorBlock &U);
    
    MatrixXcd FullOperator();

    void CheckConsistency();
    void PrintInformation();
};

class WavefunctionBlock
{
public:
    int quantumN_sector;
    vector<MatrixXcd> block;
    vector<int> QuantumN;
    
    WavefunctionBlock();
    WavefunctionBlock(int _size);
    
    int SearchQuantumN(int n) const;
    
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
    
    WavefunctionBlock operator*(complex<double> n);
    WavefunctionBlock operator/(complex<double> n);
    WavefunctionBlock& operator*=(complex<double> n);
    WavefunctionBlock& operator/=(complex<double> n);

    
    void Truncate(OperatorBlock& U, BlockPosition pos, bool transposed);
};

class DMRGBlock
{
public:
    OperatorBlock H;
    OperatorBlock U;

    vector<SuperBlock> c_up;
    vector<SuperBlock> c_down;
    
    //vector<OperatorBlock> n_up;
    //vector<OperatorBlock> n_down;
    
    
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
        
        //n_up.resize(n);
        //n_down.resize(n);
        
        return *this;
    }
    
    size_t size()
    {
        return c_up.size();
    }
    
    void UpdateQN(const vector<int>& qn, int _size);

};


#endif /* Class_DMRGBlock_hpp */
