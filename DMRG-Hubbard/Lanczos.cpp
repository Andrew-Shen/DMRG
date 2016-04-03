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
#include "U(1)_Symmetry.hpp"


#include <iostream>
#include <vector>
#include <Eigen/Eigenvalues>

using namespace Eigen;
using namespace std;


WavefunctionBlock random_wavefunction(DMRGSystem &S, int n)
{
    int left_size = S.left_size;
    int right_size = S.right_size;
    
    VectorXd number_left =  S.BlockL[left_size].number;
    VectorXd number_right = S.BlockR[right_size].number;
    int n_max_left = (int)number_left.size() - 1;
    int n_max_right = (int)number_right.size() - 1;
    
    WavefunctionBlock npsi(n);
    for (int i = 0; i <= n; i++) {
        if ((n - i > n_max_right) || (i > n_max_left)) {
            npsi.block[i].resize(0, 0);
            continue;
        }
        if (S.BlockL[left_size].number(i) == 0 || S.BlockR[right_size].number(n - i) == 0) {
            npsi.block[i].resize(0, 0);
        } else {
            npsi.block[i] = npsi.block[i] = MatrixXd::Random(S.BlockL[left_size].number(i), S.BlockR[right_size].number(n - i));
        }
    }
    
    return npsi;
}

WavefunctionBlock zero_wavefunction(DMRGSystem &S, int n)
{
    int left_size = S.left_size;
    int right_size = S.right_size;
    
    VectorXd number_left =  S.BlockL[left_size].number;
    VectorXd number_right = S.BlockR[right_size].number;
    int n_max_left = (int)number_left.size() - 1;
    int n_max_right = (int)number_right.size() - 1;
    
    WavefunctionBlock npsi(n);
    for (int i = 0; i <= n; i++) {
        if ((n - i > n_max_right) || (i > n_max_left)) {
            npsi.block[i].resize(0, 0);
            continue;
        }
        if (S.BlockL[left_size].number(i) == 0 || S.BlockR[right_size].number(n - i) == 0) {
            npsi.block[i].resize(0, 0);
        } else {
            npsi.block[i] = MatrixXd::Zero(S.BlockL[left_size].number(i), S.BlockR[right_size].number(n - i));
        }
    }
    
    return npsi;
}

WavefunctionBlock one_wavefunction(DMRGSystem &S, int n)
{
    int left_size = S.left_size;
    int right_size = S.right_size;
    
    VectorXd number_left =  S.BlockL[left_size].number;
    VectorXd number_right = S.BlockR[right_size].number;
    int n_max_left = (int)number_left.size() - 1;
    int n_max_right = (int)number_right.size() - 1;
    
    WavefunctionBlock npsi(n);
    for (int i = 0; i <= n; i++) {
        if ((n - i > n_max_right) || (i > n_max_left)) {
            npsi.block[i].resize(0, 0);
            continue;
        }
        if (S.BlockL[left_size].number(i) == 0 || S.BlockR[right_size].number(n - i) == 0) {
            npsi.block[i].resize(0, 0);
        } else {
            npsi.block[i] = MatrixXd::Ones(S.BlockL[left_size].number(i), S.BlockR[right_size].number(n - i));
        }
        
        //cout << "npsi.block" << i << endl;
        //cout << npsi.block[i] << endl;
    }
    
    return npsi;
}

double inner_product(WavefunctionBlock &v1, WavefunctionBlock &v2)
{
    assert(v1.total_particle_number == v2.total_particle_number && "Particle numbers do not match! ");
    
    MatrixXd tmat;
    double res = 0;
    
    for (int i = 0; i <= v1.total_particle_number; i++) {
        //if (v1.block[i].cols() != v2.block[i].cols()) {
        //    cout << v1.block[i].cols() << "kk"<< v2.block[i].cols() << endl;
        //    cout << v1.block[i].rows() << "kk"<< v2.block[i].rows() << endl;
        //}
        assert(v1.block[i].cols() == v2.block[i].cols() && "Matrix incosistent! ");
        assert(v1.block[i].rows() == v2.block[i].rows() && "Matrix incosistent! ");
        
        tmat = v1.block[i].transpose() * v2.block[i];
        res += tmat.trace();
    }
    
    return res;
    
}

double Lanczos(DMRGSystem &S, int n, int _max_iter, double _rel_err, bool have_seed)
{
    int max_iter;       // Max number of Lanczos iteration
    double rel_err;
    vector<double> es;
    //vector<MatrixXd> v; // v in Lanczos algorithm
    //vector<MatrixXd> w; // w in Lanczos algorithm
    //vector<MatrixXd> wp;    // w' in Lanczos algorithm
    vector<WavefunctionBlock> v;
    
    vector<WavefunctionBlock> w;
    vector<WavefunctionBlock> wp;
    VectorXd main_diag; // Main diagonal in the tridiagonal matrix
    VectorXd super_diag;// First super diagonal in the tridiagonal matrix
    MatrixXd tridiag;   // tridiagonal matrix given by Lanczos algorithm
    
    int left_size = S.left_size;
    int right_size = S.right_size;
    int dim_l = S.BlockL[left_size].number.sum();
    int dim_r = S.BlockR[right_size].number.sum();

    size_t left_n_size = S.BlockL[left_size].number.size();
    size_t right_n_size = S.BlockR[right_size].number.size();
    if (left_n_size > right_n_size) {
        S.BlockR[right_size].number.conservativeResize(left_n_size);
        for (int i = 0; i < left_n_size - right_n_size; i++) {
            S.BlockR[right_size].number(right_n_size + i) = 0;
        }
    }
    
    if (right_n_size > left_n_size) {
        S.BlockL[left_size].number.conservativeResize(right_n_size);
        for (int i = 0; i < right_n_size - left_n_size; i++) {
            S.BlockL[left_size].number(left_n_size + i) = 0;
        }
    }
    
    //cout << S.BlockL[left_size].number.transpose() << endl;
    //cout << S.BlockR[right_size].number.transpose() << endl;
    //assert(S.BlockL[left_size].number.size() == S.BlockR[right_size].number.size());
    
    
    max_iter = min(_max_iter, dim_l * dim_r);
    rel_err = _rel_err;
    
    // Initialization of Lanczos
    es.resize(max_iter + 1);
    v.resize(max_iter + 1);
    w.resize(max_iter + 1);
    wp.resize(max_iter + 1);
    /*
    for (int i = 0; i < v.size(); i++) {
        v[i].resize(n);
        w[i].resize(n);
        wp[i].resize(n);
    }
     */
    main_diag.resize(max_iter + 1);
    main_diag = VectorXd::Zero(max_iter + 1);
    super_diag.resize(max_iter + 1);
    super_diag = VectorXd::Zero(max_iter + 1);
    v[0] = zero_wavefunction(S, n);
    
    // Wavefunction Prediction
    if (have_seed == true) {
        v[1] = S.seed;
    } else {
        v[1] = one_wavefunction(S, n);
        //v[1] = random_wavefunction(S, n);
    }
    v[1].normalize();
    
    SelfAdjointEigenSolver<MatrixXd> tsolver;
    S.psi = zero_wavefunction(S, n);
    for (int i = 1; i < max_iter; i++) {
        //cout << "Iteration: " << i << endl;
        wp[i] = symmetric_prod(S, n, v[i]);
        
        //cout << wp[i].block[0].size() << endl;
        
        //for (int j = 0; j <= wp[i].total_particle_number; j++) {
        //    cout << "wp(" << j << ")=" << endl;
        //    cout << wp[i].block[j] << endl;
            //cout << "v(" << j << ")=" << endl;
            //cout << v[i].block[j]<< endl;
        //}
        //cout << inner_product(wp[i], v[i]) << endl;
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
            cout << "main_diag " << main_diag.transpose() << endl;
            cout << "super_diag " << super_diag.transpose() << endl;
            cout << tridiag << endl;
            abort();
        }
        es[i - 1] = tsolver.eigenvalues()(0);
        
        if (absl(es[i - 1] - es[i - 2]) < rel_err ) {
            for (int j = 0; j < i; j++) {
            S.psi = S.psi + v[j + 1] * tsolver.eigenvectors()(j,0);
            }
            cout << "Lancozs iteration: " << i << endl;
            return es[i - 1];
            break;
        }

    }
    
    tridiag.resize(max_iter, max_iter);
    tridiag = MatrixXd::Zero(max_iter, max_iter);
    w[max_iter] = symmetric_prod(S, n, v[max_iter]);
    
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


WavefunctionBlock symmetric_prod(DMRGSystem &S, int n, WavefunctionBlock &psi)
{
    int left_size = S.left_size;
    int right_size = S.right_size;

    VectorXd number_left =  S.BlockL[left_size].number;
    VectorXd number_right = S.BlockR[right_size].number;
    int n_max_left = (int)number_left.size() - 1;
    int n_max_right = (int)number_right.size() - 1;
    
    WavefunctionBlock npsi(n);
    for (int i = 0; i <= n; i++) {
        //cout << n - i << endl;
        if ((n - i > n_max_right) || (i > n_max_left)) {
            npsi.block[i].resize(0, 0);
            continue;
        }
        if (S.BlockL[left_size].number(i) == 0 || S.BlockR[right_size].number(n - i) == 0) {
            npsi.block[i].resize(0, 0);
            continue;
        }
        npsi.block[i] = MatrixXd::Zero(S.BlockL[left_size].number(i), S.BlockR[right_size].number(n - i));
    }

    MatrixXd tmat;
    //cout << n << endl;
    //cout << n_max_left << endl;
    for (int i = 0; i <= min(n_max_left, n); i++) {
        if (n - i > n_max_right) {
            //cout << "n - i > n_max_right" << endl;
            continue;
        }
        if (npsi.block[i].cols() == 0 || npsi.block[i].rows() == 0) {
            //cout << "b" << endl;
            continue;
        }
        //cout << "i=" << i << endl;
        // H_L H_R
        //cout << psi.block[i] << endl;
        //cout << S.BlockL[left_size].H.block[i] << endl;
        
        assert(S.BlockL[left_size].H.block[i].cols() != 0 && S.BlockR[right_size].H.block[n - i].cols() != 0);
        tmat = S.BlockL[left_size].H.block[i] * psi.block[i];
        npsi.block[i] += tmat * S.BlockR[right_size].H.block[n - i];
        
        //cout << "kh" << i << endl;

        
        // c_L^dag c_R
        if (n - i > 0) {
            if (S.BlockL[left_size].number(i) != 0 && S.BlockR[right_size].number(n - i - 1) != 0 &&
                i != n_max_left) {
                //cout << "i = " << i << " n-i-1 = " << n - i -1 << endl;
                //cout << S.BlockL[left_size].number(i) << " " << S.BlockR[right_size].number(n - i - 1) << endl;
                //cout << S.BlockL[left_size].c_up[left_size].block[i].rows() << " " << S.BlockR[right_size].c_up[right_size].block[n - i - 1].rows() << endl;
                
                if (S.BlockL[left_size].number(i + 1) != 0) {
                    assert(S.BlockL[left_size].c_up[left_size].block[i].size() != 0 &&
                           S.BlockR[right_size].c_up[right_size].block[n - i - 1].size() != 0);
                    
                    tmat = S.BlockL[left_size].c_up[left_size].block[i].transpose() * psi.block[i];
                    npsi.block[i + 1] += tmat * S.BlockR[right_size].c_up[right_size].block[n - i - 1].transpose();
                    tmat = S.BlockL[left_size].c_down[left_size].block[i].transpose() * psi.block[i];
                    npsi.block[i + 1] += tmat * S.BlockR[right_size].c_down[right_size].block[n - i - 1].transpose();
                    
                    //cout << "kldag r" << endl;
                
                }
            }
        }
        // c_L c_R^dag
        if (i > 0) {
            //cout << S.BlockL[left_size].number(i - 1) << " " << S.BlockR[right_size].number(n - i) << " " << n_max_right << endl;
            if (S.BlockL[left_size].number(i - 1) != 0 && S.BlockR[right_size].number(n - i) != 0 &&
                (n - i) != n_max_right) {
                //cout << "a" << endl;
                if (S.BlockR[right_size].number(n - i + 1) != 0) {
                    //cout << "i - 1 = " << i - 1<< " n- i = " << n - i << " n_max_right" << n_max_right << endl;
                    //cout << S.BlockL[left_size].number(i- 1) << " " << S.BlockR[right_size].number(n - i ) << endl;
                    //cout << S.BlockL[left_size].c_up[left_size].block[i - 1].rows() << " " << S.BlockR[right_size].c_up[right_size].block[n - i ].rows() << endl;
                    assert(S.BlockL[left_size].c_up[left_size].block[i - 1].size() != 0 &&
                           S.BlockR[right_size].c_up[right_size].block[n - i].size() != 0);
                    
                    tmat = S.BlockL[left_size].c_up[left_size].block[i - 1] * psi.block[i];
                    npsi.block[i - 1] += tmat * S.BlockR[right_size].c_up[right_size].block[n - i];
                    //cout << S.BlockL[left_size].c_down[left_size].block[i - 1] << endl;
                    //cout << "psi" << endl;
                    //cout << S.BlockL[left_size].c_down[left_size].block[i] << endl;
                    tmat = S.BlockL[left_size].c_down[left_size].block[i - 1] * psi.block[i];
                    npsi.block[i - 1] += tmat * S.BlockR[right_size].c_down[right_size].block[n - i];
                    
                    //cout << "kl r dag" << endl;
                }
                
            }
        }
    }
    
    //for (int i = 0; i <= n; i++) {
    //    cout << "npsi block " << i << endl;
    //    cout << npsi.block[i] << endl;
    //}
    
    //cout << "kkkkkkkk" << endl;
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