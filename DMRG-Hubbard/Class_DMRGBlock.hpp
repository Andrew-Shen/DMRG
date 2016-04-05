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

MatrixXd matrix_direct_plus(MatrixXd &m1, MatrixXd &m2);
void matrix_reorder(MatrixXd &m, vector<int> &vec_idx);
vector<int> sort_index(vector<int> &vec);

class OperatorBlock
{
public:
    vector<MatrixXd> block;
    vector<int> QuantumN;
    
    OperatorBlock();
    OperatorBlock(int _size);
    
    OperatorBlock &resize(int n);
    
    size_t size()
    {
        return QuantumN.size();
    }
    
    void CheckConsistency();
    void ZeroPurification();
    void Update(MatrixXd &m, vector<int> &qn);
    
    MatrixXd Operator_full();
    vector<int> QuantumN_full();
};

vector<int> QuantumN_kron(OperatorBlock &ob1, OperatorBlock &ob2);




class WavefunctionBlock
{
public:
    int total_particle_number;
    
    vector<MatrixXd> block;
    
    WavefunctionBlock();
    WavefunctionBlock(int _size);
    
    WavefunctionBlock& resize(int n);
    
    double norm();
    WavefunctionBlock& normalize();

    
    WavefunctionBlock operator*(double n);
    WavefunctionBlock operator/(double n);
    WavefunctionBlock operator+(WavefunctionBlock x);
    WavefunctionBlock operator-(WavefunctionBlock x);
};


class DMRGBlock
{
public:
    int size;
    VectorXd number;
    
    OperatorBlock H;
    OperatorBlock U;

    vector<OperatorBlock> c_up;
    vector<OperatorBlock> c_down;
    //vector<VectorXd> number;
    

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
        size = n;
        
        c_up.resize(n);
        c_down.resize(n);
        //number.resize(n);
        
        
        return *this;
    }
};


#endif /* Class_DMRGBlock_hpp */
