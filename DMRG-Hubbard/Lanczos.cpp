//
//  Lanczos.cpp
//  DMRG-Hubbard
//
//  Created by Andrew Shen on 16/3/26.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//


#include "Lanczos.hpp"
#include "Class_DMRGSystem.hpp"
#include "Class_DMRGBlock.hpp"


#include <iostream>
#include <vector>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;

WavefunctionBlock InitializeWavefunction(const vector<int>& quantumN_left, const vector<int>& quantumN_right,
                                       const vector<size_t>& block_size_left, const vector<size_t>& block_size_right,
                                       int n, WBType wb_type)
{
    WavefunctionBlock npsi;
    npsi.quantumN_sector = n;
    npsi.block.clear();
    npsi.QuantumN.clear();
    
    for (int i = 0; i < quantumN_left.size(); i++) {
        int qn_left = quantumN_left[i];
        if (qn_left > n) {
            break;
        }
        int block_idx_right = SearchIndex(quantumN_right, n - qn_left);
        if (block_idx_right == -1) {
            continue;
        }
        npsi.QuantumN.push_back(qn_left);
        switch (wb_type) {
            case WBType::RANDOM:
                npsi.block.push_back(MatrixXd::Random(block_size_left[i], block_size_right[block_idx_right]));
                break;
            case WBType::ONES:
                npsi.block.push_back(MatrixXd::Ones(block_size_left[i], block_size_right[block_idx_right]));
                break;
            case WBType::ZERO:
                npsi.block.push_back(MatrixXd::Zero(block_size_left[i], block_size_right[block_idx_right]));
                break;
            default:
                break;
        }
    }
    if (wb_type != WBType::ZERO) {
        npsi.normalize();
    }
    return npsi;
}

WavefunctionBlock InitializeWavefunction(const WavefunctionBlock& seed, WBType wb_type)
{
    WavefunctionBlock npsi(seed);
    
    for (int i = 0; i < seed.size(); i++) {
        switch (wb_type) {
            case WBType::RANDOM:
                npsi.block[i] = MatrixXd::Random(seed.block[i].rows(),
                                                 seed.block[i].cols());
                break;
            case WBType::ONES:
                npsi.block[i] = MatrixXd::Ones(seed.block[i].rows(),
                                               seed.block[i].cols());
                break;
            case WBType::ZERO:
                npsi.block[i] = MatrixXd::Zero(seed.block[i].rows(),
                                               seed.block[i].cols());
                break;
            default:
                break;
        }
    }
    if (wb_type != WBType::ZERO) {
        npsi.normalize();
    }
    return npsi;
}

double inner_product(WavefunctionBlock &v1, WavefunctionBlock &v2)
{
    assert(v1.quantumN_sector == v2.quantumN_sector && "WavefunctionBlock InnerProduct: Quantum numbers do not match! ");
    
    MatrixXd tmat;
    double res = 0;
    
    for (int i = 0; i < v1.size(); i++) {
        assert(v1.block[i].cols() == v2.block[i].cols() && "WavefunctionBlock InnerProduct: Matrix incosistent! ");
        assert(v1.block[i].rows() == v2.block[i].rows() && "WavefunctionBlock InnerProduct: Matrix incosistent! ");
        
        tmat = v1.block[i].transpose() * v2.block[i];
        res += tmat.trace();
    }
    
    return res;
}

double Lanczos(DMRGSystem &S, int n, int _max_iter, double _rel_err)
{
    int max_iter;       // Max number of Lanczos iteration
    double rel_err;
    vector<double> es;

    vector<WavefunctionBlock> v;
    vector<WavefunctionBlock> w;
    vector<WavefunctionBlock> wp;
    VectorXd main_diag; // Main diagonal in the tridiagonal matrix
    VectorXd super_diag;// First super diagonal in the tridiagonal matrix
    MatrixXd tridiag;   // tridiagonal matrix given by Lanczos algorithm
    
    int left_size = S.left_size;
    int right_size = S.right_size;
    size_t dim_l = S.BlockL[left_size].H.total_d();
    size_t dim_r = S.BlockR[right_size].H.total_d();
    
    max_iter = min(_max_iter, (int)(dim_l * dim_r));
    rel_err = _rel_err;
    
    // Initialization of Lanczos
    es.resize(max_iter + 1);
    v.resize(max_iter + 1);
    w.resize(max_iter + 1);
    wp.resize(max_iter + 1);
 
    main_diag.resize(max_iter + 1);
    main_diag = VectorXd::Zero(max_iter + 1);
    super_diag.resize(max_iter + 1);
    super_diag = VectorXd::Zero(max_iter + 1);
    
    v[0] = InitializeWavefunction(S.seed, WBType::ZERO);
    v[1] = S.seed;
    
    S.psi = InitializeWavefunction(S.seed, WBType::ZERO);
    
    SelfAdjointEigenSolver<MatrixXd> tsolver;
    for (int i = 1; i < max_iter; i++) {
        wp[i] = symmetric_prod(S, v[i]);
        
        main_diag(i - 1) = inner_product(wp[i], v[i]);

        if (i == 1) {
            w[i] = wp[i] - v[i] * main_diag(i - 1);
        } else {
            w[i] = wp[i] - v[i] * main_diag(i - 1) - v[i - 1] * super_diag(i - 2);
        }
        super_diag(i - 1) = w[i].norm();
        v[i + 1] = w[i] / super_diag(i - 1);
        tridiag = MatrixXd::Zero(i, i);

        if (i == 1) {
            es[i - 1] = main_diag(i - 1);
            continue;
        }
        for (int j = 0; j < i - 1; j++) {
            tridiag(j, j) = main_diag(j);
            tridiag(j, j + 1) = super_diag(j);
        }
        tridiag(i - 1, i - 1) = main_diag(i - 1);
        tridiag.diagonal(-1) = tridiag.diagonal(1);
        tsolver.compute(tridiag);
        if (tsolver.info() != Success) {
            abort();
        }
        es[i - 1] = tsolver.eigenvalues()(0);
        
        if (absl(es[i - 1] - es[i - 2]) < rel_err ) {
            for (int j = 0; j < i; j++) {
            S.psi += v[j + 1] * tsolver.eigenvectors()(j,0);
            }
            cout << "Lancozs iteration: " << i << endl;
            
            return es[i - 1];
            break;
        }

    }
    
    tridiag.resize(max_iter, max_iter);
    tridiag = MatrixXd::Zero(max_iter, max_iter);
    w[max_iter] = symmetric_prod(S, v[max_iter]);
    
    main_diag(max_iter - 1) = inner_product(w[max_iter], v[max_iter]);
    for (int j = 0; j < max_iter - 2; j++) {
        tridiag(j, j) = main_diag(j);
        tridiag(j, j + 1) = super_diag(j);
    }
    tridiag(max_iter - 1, max_iter - 1) = main_diag(max_iter - 1);
    tridiag.diagonal(-1) = tridiag.diagonal(1);
    tsolver.compute(tridiag);
    if (tsolver.info() != Success) abort();
    es[max_iter - 1] = tsolver.eigenvalues()(0);
    
    for (int j = 0; j < max_iter; j++) {
        S.psi = S.psi + v[j + 1] * tsolver.eigenvectors()(j,0);
    }
    cout << "Max number of iteration reaches! Final Error: " <<  absl(es[max_iter - 1] - es[max_iter - 2]) << endl;
    return es[max_iter - 1];
    
}

WavefunctionBlock symmetric_prod(DMRGSystem &S, WavefunctionBlock &psi)
{
    int left_size = S.left_size;
    int right_size = S.right_size;

    int n = psi.quantumN_sector;
    
    // implement in the class ?
    WavefunctionBlock npsi(psi);
    npsi.block.clear();
    
    MatrixXd tmat;

    int left_qn, left_idx, right_idx;
    
    // H_L H_R
    for (int i = 0; i < psi.size(); i++) {
        left_qn = psi.QuantumN[i];
        left_idx = S.BlockL[left_size].H.SearchQuantumN(left_qn);
        right_idx = S.BlockR[right_size].H.SearchQuantumN(n - left_qn);
        
        tmat = S.BlockL[left_size].H.block[left_idx] * psi.block[i];
        tmat += psi.block[i] * S.BlockR[right_size].H.block[right_idx].transpose();
        npsi.block.push_back(tmat);
    }
    
    // c_L^dag c_R
    int wb_idx;
    for (int i = 0; i < psi.size(); i++) {
        left_qn = psi.QuantumN[i];
        left_idx = S.BlockL[left_size].H.SearchQuantumN(left_qn);
        right_idx = S.BlockR[right_size].H.SearchQuantumN(n - left_qn - 1);
        wb_idx = psi.SearchQuantumN(left_qn + 1);
        
        if (left_idx == -1 || right_idx == -1 || wb_idx == -1) {
            continue;
        }
        
        tmat = S.BlockL[left_size].c_up[left_size].block[left_idx].transpose() * psi.block[i];
        npsi.block[wb_idx] += tmat * S.BlockR[right_size].c_up[right_size].block[right_idx].transpose();
        tmat = S.BlockL[left_size].c_down[left_size].block[left_idx].transpose() * psi.block[i];
        npsi.block[wb_idx] += tmat * S.BlockR[right_size].c_down[right_size].block[right_idx].transpose();
    }
    
    // c_L c_R^dag
    for (int i = 0; i < psi.size(); i++) {
        left_qn = psi.QuantumN[i];
        left_idx = S.BlockL[left_size].H.SearchQuantumN(left_qn - 1);
        right_idx = S.BlockR[right_size].H.SearchQuantumN(n - left_qn);
        wb_idx = psi.SearchQuantumN(left_qn - 1);
        
        if (left_idx == -1 || right_idx == -1 || wb_idx == -1) {
            continue;
        }
        
        tmat = S.BlockL[left_size].c_up[left_size].block[left_idx] * psi.block[i];
        npsi.block[wb_idx] += tmat * S.BlockR[right_size].c_up[right_size].block[right_idx];
        tmat = S.BlockL[left_size].c_down[left_size].block[left_idx] * psi.block[i];
        npsi.block[wb_idx] += tmat * S.BlockR[right_size].c_down[right_size].block[right_idx];
    }
    
    return npsi;
}

double absl(double _a)
{
    if (_a > 0) {
        return _a;
    } else {
        return -_a;
    }
}