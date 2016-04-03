//
//  Class_DMRGBlock.cpp
//  DMRG-Heisenberg
//
//  Created by Andrew Shen on 16/3/27.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//


#include "Class_DMRGBlock.hpp"
#include "U(1)_Symmetry.hpp"


#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <math.h>

using namespace Eigen;
using namespace std;

/*
DMRGBlock &DMRGBlock::resize(int n)
{
    size = n;
    sz.resize(n);
    splus.resize(n);
    return *this;
}

DMRGBlock::DMRGBlock()
{
    resize(1);
}

DMRGBlock::DMRGBlock(int _size)
{
    resize(_size);
}

*/

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
    max_particle_number = n;
    
    block.resize(n + 1);
    
    return *this;
}

MatrixXd OperatorBlock::regenerate()
{
    MatrixXd tmat;
    
    for (int i = 0; i <= max_particle_number; i++) {
        tmat.noalias() = matrix_direct_plus(tmat, block[i]);
    }
    
    return tmat;
    
}

MatrixXd OperatorBlock::super_regenerate()
{
    MatrixXd tmat;
    MatrixXd tmat2;
    
    int first_block_row = 0;
    int last_block_col = 0;
    
    for (int i = 0; i <= max_particle_number; i++) {
        tmat.noalias() = matrix_direct_plus(tmat, block[i]);
        if (first_block_row == 0 && block[i].rows() != 0) {
            first_block_row = (int)block[i].rows();
        }
        if (block[i].cols() != 0) {
            last_block_col = (int)block[i].cols();
        }
        
    }
    
    tmat2 = MatrixXd::Zero(tmat.rows() + last_block_col, tmat.cols() + first_block_row);
    tmat2.block(0, first_block_row, tmat.rows(), tmat.cols()) = tmat;
    
    return tmat2;
    
}

void OperatorBlock::save_block(MatrixXd &m, VectorXd &v)
{
    int dm = (int)v.sum();
    assert((m.cols() == dm) && "Save Matrix to Block: Dimensions do not agree! ");
    
    this -> resize((int)v.size() - 1);
    
    int begin_pos;
    int range;
    for (int i = 0; i <= max_particle_number; i++) {
        if (v(i) == 0) {
            block[i].resize(0, 0);
            continue;
        }
        block[i].resize(v(i), v(i));

        
        begin_pos = block_begin(v, i);
        range = block_end(v, i) - block_begin(v, i) + 1;    // Be careful with the additional 1 here.
        block[i] = m.block(begin_pos, begin_pos, range, range);
        
        //cout << "H.block[" << i << "]=" << endl;
        //cout << block[i] << endl;
    }
}

void OperatorBlock::save_superblock(MatrixXd &m, VectorXd &v)
{
    int dm = (int)v.sum();
    assert((m.cols() == dm) && "Save Matrix to Superblock: Dimensions do not agree! ");
    
    this -> resize((int)v.size() - 1);
    
    int begin_pos;
    int range;
    for (int i = 0; i < max_particle_number; i++) {
        if (v(i) == 0) {
            block[i].resize(0, 0);
            continue;
        }
        block[i].resize(v(i), v(i + 1));
        
        begin_pos = block_begin(v, i);
        range = block_end(v, i) - block_begin(v, i) + 1;

        block[i] = m.block(begin_pos, begin_pos + range, range, block_end(v, i + 1) - block_begin(v, i + 1) + 1);
        //cout << "i=" << i << endl;
        //cout << block[i] << endl;
    }
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
