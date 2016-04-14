//
//  Class_DMRGSystem.cpp
//  DMRG-Hubbard
//
//  Created by Andrew Shen on 16/3/26.
//  Copyright © 2016 Andrew Shen. All rights reserved.
//


#include "Class_DMRGSystem.hpp"
#include "Lanczos.hpp"

#include <iostream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <algorithm>

using namespace Eigen;
using namespace std;


DMRGSystem::DMRGSystem(int _nsites, int _max_lanczos_iter, double _rel_err, double u)
{
    nsites = _nsites;
    
    BlockL.resize(nsites);
    BlockR.resize(nsites);

    max_lanczos_iter = _max_lanczos_iter;
    rel_err = _rel_err;

    d_per_site = 4; // Dimension of Hilbert space per site.
    
    c_up0 = MatrixXd::Zero(d_per_site, d_per_site);
    c_down0 = MatrixXd::Zero(d_per_site, d_per_site);
    u0 = MatrixXd::Zero(d_per_site, d_per_site);
    
    sz0 = MatrixXd::Zero(d_per_site, d_per_site);
    n_up0 = MatrixXd::Zero(d_per_site, d_per_site);
    n_down0 = MatrixXd::Zero(d_per_site, d_per_site);

    c_up0(0,1) = 1;
    c_up0(2,3) = 1;
    c_down0(0,2) = 1;
    c_down0(1,3) = 1;
    u0(3,3) = 1;
    u0 = u * u0;

    sz0(1,1) = 0.5;
    sz0(2,2) = -0.5;
    n_up0(1,1) = 1.0;
    n_up0(3,3) = 1.0;
    n_down0(2,2) = 1.0;
    n_down0(3,3) = 1.0;
    
    vector<int> tsize = {0, 1, 1, 2};
    H0.Update(u0, tsize);
    BlockL[0].H = H0;
    BlockR[0].H = H0;
    BlockL[0].c_up[0].Update(c_up0, tsize);
    BlockL[0].c_down[0].Update(c_down0, tsize);
    BlockR[0].c_up[0].Update(c_up0, tsize);
    BlockR[0].c_down[0].Update(c_down0, tsize);
}

void DMRGSystem::BuildBlockLeft(int _iter)
{
    left_size = _iter;
    
    BlockL[left_size].resize(left_size + 1);
    
    MatrixXd HL = BlockL[left_size - 1].H.Operator_full();
    MatrixXd c_upL = BlockL[left_size - 1].c_up[left_size - 1].Operator_full();
    MatrixXd c_downL = BlockL[left_size - 1].c_down[left_size - 1].Operator_full();
    
    size_t dim_l = HL.cols();
    
    assert(dim_l == c_upL.cols() && "BuildBlock: Dimensions are not properly set in the last iteration! ");
    
    MatrixXd I_left = MatrixXd::Identity(dim_l, dim_l);
    MatrixXd I_site = MatrixXd::Identity(d_per_site, d_per_site);

    MatrixXd H = kroneckerProduct(HL, I_site) + kroneckerProduct(I_left, u0) +
    kroneckerProduct(c_upL, c_up0.transpose()) + kroneckerProduct(c_upL.transpose(), c_up0) +
    kroneckerProduct(c_downL, c_down0.transpose()) + kroneckerProduct(c_downL.transpose(), c_down0);
    
    vector<int> quantumN = QuantumN_kron(BlockL[left_size - 1].H, H0);
    vector<int> trans_idx = sort_index(quantumN);
    
    matrix_reorder(H, trans_idx);
    BlockL[left_size].H.Update(H, quantumN);
    
    c_upL = kroneckerProduct(I_left, c_up0);
    matrix_reorder(c_upL, trans_idx);
    BlockL[left_size].c_up[left_size].Update(c_upL, quantumN);
    
    c_downL = kroneckerProduct(I_left, c_down0);
    matrix_reorder(c_downL, trans_idx);
    BlockL[left_size].c_down[left_size].Update(c_downL, quantumN);
}


void DMRGSystem::BuildBlockRight(int _iter)
{
    right_size = _iter;
    
    BlockR[right_size].resize(right_size + 1);
    
    MatrixXd HR = BlockR[right_size - 1].H.Operator_full();
    MatrixXd c_upR = BlockR[right_size - 1].c_up[right_size - 1].Operator_full();
    MatrixXd c_downR = BlockR[right_size - 1].c_down[right_size - 1].Operator_full();
    
    size_t dim_r = HR.cols();
    
    assert(dim_r == c_upR.cols() && "BuildBlock: Dimensions are not properly set in the last iteration! ");
    
    MatrixXd I_right = MatrixXd::Identity(dim_r, dim_r);
    MatrixXd I_site = MatrixXd::Identity(d_per_site, d_per_site);
    
    MatrixXd H = kroneckerProduct(I_site, HR) + kroneckerProduct(u0, I_right) +
    kroneckerProduct(c_up0, c_upR.transpose()) + kroneckerProduct(c_up0.transpose(), c_upR) +
    kroneckerProduct(c_down0, c_downR.transpose()) + kroneckerProduct(c_down0.transpose(), c_downR);
    
    vector<int> quantumN = QuantumN_kron(H0, BlockR[right_size - 1].H);
    vector<int> trans_idx = sort_index(quantumN);
    
    matrix_reorder(H, trans_idx);
    BlockR[right_size].H.Update(H, quantumN);
    
    
    c_upR = kroneckerProduct(c_up0, I_right);
    matrix_reorder(c_upR, trans_idx);
    BlockR[right_size].c_up[right_size].Update(c_upR, quantumN);
    
    c_downR = kroneckerProduct(c_down0, I_right);
    matrix_reorder(c_downR, trans_idx);
    BlockR[right_size].c_down[right_size].Update(c_downR, quantumN);
}


void DMRGSystem::GroundState(int n, bool wf_prediction)
{
    double ev;
        
    ev = Lanczos(*this, n, max_lanczos_iter, rel_err, wf_prediction);
    
    cout << "Energy per site: " << std::setprecision(12) << ev / n  << endl;
}



double DMRGSystem::Truncate(BlockPosition _position, int _max_m, double _trun_err)
{
    int n = psi.quantumN_sector;
    size_t n_blocks = psi.QuantumN.size();
    
    MatrixXd tmat;
    
    // implement in a class function?
    rho.QuantumN.clear();
    rho.block.clear();
    if (_position == BlockPosition::LEFT) {
        for (int i = 0; i < n_blocks; i++) {
            rho.QuantumN.push_back(psi.QuantumN[i]);
            rho.block.push_back(psi.block[i] * psi.block[i].transpose());
        }
    } else {
        for (int i = 0; i < n_blocks; i++) {
            // Notice the i index is denoted as the quantum number of the LEFT block.
            // Thus for manipulations of RIGHT block the index should be reversed.
            
            rho.QuantumN.push_back(n - psi.QuantumN[i]);
            rho.block.push_back(psi.block[i].transpose() * psi.block[i]);
        }
        // is this really necessary?
        reverse(rho.QuantumN.begin(), rho.QuantumN.end());
        reverse(rho.block.begin(), rho.block.end());
        
        //rho.PrintInformation();
    }
    
    size_t total_d = rho.total_d();
    
    assert(n_blocks == rho.size() && "Truncate: Dimension of psi and rho do not match! ");

    vector<MatrixXd> rho_evec(n_blocks);
    vector<double> rho_eig_t, tvec;
    vector<int> eig_idx;
    
    SelfAdjointEigenSolver<MatrixXd> rsolver;
    for (int i = 0; i < n_blocks; i++) {
        rsolver.compute(-rho.block[i]);
        if (rsolver.info() != Success) abort();
        
        tvec.resize(rsolver.eigenvalues().size());
        VectorXd::Map(&tvec[0], rsolver.eigenvalues().size()) = -rsolver.eigenvalues();
        rho_eig_t.insert(rho_eig_t.end(), tvec.begin(), tvec.end());
        
        rho_evec[i] = rsolver.eigenvectors();
    }
    
    for (int i = 0; i < total_d; i++) {
        rho_eig_t.at(i) = -rho_eig_t.at(i);
    }
    eig_idx = sort_index_double(rho_eig_t);
    for (int i = 0; i < total_d; i++) {
        rho_eig_t.at(i) = -rho_eig_t.at(i);
    }
    
    error = 0;

    int _m = 0;
    double inv_error = 0;
    _max_m = min(_max_m, (int)rho_eig_t.size());
    for (int i = 0; i < _max_m; i++) {
        inv_error += rho_eig_t.at(i);
        if ((1 - inv_error) < _trun_err) {
            _m = i + 1;
            break;
        }
    }
    error = 1 - inv_error;
    if (_m == 0) {
        _m = _max_m;
        cout << "Maximum truncation number reaches. " << endl;
    }
    cout << "Truncate at " << _m << " states. Error = " << error << endl;
    
    vector<bool> truncation_flag(n_blocks);
    for (int i = 0; i < n_blocks; i++) {
        truncation_flag[i] = false;
    }
    int block_m;
    
    for (int i = _m; i < total_d; i++) {
        for (int j = 0; j < n_blocks; j++) {
            if ((eig_idx.at(i) <= rho.end(j)) && (eig_idx.at(i) >= rho.begin(j))) {
                if (truncation_flag[j] != true) {
                    block_m = eig_idx[i] - rho.begin(j);
                    tmat = rho_evec[j].leftCols(block_m);
                    rho_evec[j] = tmat;
                    truncation_flag[j] = true;
                }
                break;
            }
        }
    }
    
    if (_position == BlockPosition::LEFT) {
        BlockL[left_size].H.RhoPurification(rho);
        
        BlockL[left_size].U.block.clear();
        BlockL[left_size].U.QuantumN.clear();
        for (int i = 0; i < n_blocks; i++) {
            BlockL[left_size].U.QuantumN.push_back(BlockL[left_size].H.QuantumN[i]);
            BlockL[left_size].U.block.push_back(rho_evec[i].transpose());
            
            tmat = BlockL[left_size].H.block[i] * rho_evec[i];
            BlockL[left_size].H.block[i] = rho_evec[i].transpose() * tmat;
            
            BlockL[left_size].H.block_size[i] = rho_evec[i].cols();
        }
        BlockL[left_size].H.ZeroPurification();

        
        BlockL[left_size].c_up[left_size].RhoPurification(rho);
        BlockL[left_size].c_down[left_size].RhoPurification(rho);
        
        for (int i = 0; i < n_blocks - 1; i++) {
            assert(BlockL[left_size].c_up[left_size].QuantumN[i] == BlockL[left_size].U.QuantumN[i]);
            
            BlockL[left_size].c_up[left_size].block_size[i] = rho_evec[i].cols();
            BlockL[left_size].c_down[left_size].block_size[i] = rho_evec[i].cols();

            if (BlockL[left_size].U.QuantumN[i] + 1 == BlockL[left_size].U.QuantumN[i + 1]) {
                
                tmat = BlockL[left_size].c_up[left_size].block[i] * rho_evec[i + 1];
                BlockL[left_size].c_up[left_size].block[i] = rho_evec[i].transpose() * tmat;
                
                tmat = BlockL[left_size].c_down[left_size].block[i] * rho_evec[i + 1];
                BlockL[left_size].c_down[left_size].block[i] = rho_evec[i].transpose() * tmat;
            } else {
                BlockL[left_size].c_up[left_size].block[i].resize(0, 0);
                BlockL[left_size].c_down[left_size].block[i].resize(0, 0);
            }
        }
        // Speical treatment to the last non-coupling matrix
        BlockL[left_size].c_up[left_size].block_size[n_blocks - 1] = rho_evec[n_blocks - 1].cols();
        BlockL[left_size].c_up[left_size].block[n_blocks - 1].resize(0, 0);

        BlockL[left_size].c_down[left_size].block_size[n_blocks - 1] = rho_evec[n_blocks - 1].cols();
        BlockL[left_size].c_down[left_size].block[n_blocks - 1].resize(0, 0);
        
        BlockL[left_size].c_up[left_size].ZeroPurification();
        BlockL[left_size].c_down[left_size].ZeroPurification();
    } else {
        BlockR[right_size].H.RhoPurification(rho);
        
        BlockR[right_size].U.block.clear();
        BlockR[right_size].U.QuantumN.clear();
        for (int i = 0; i < n_blocks; i++) {
            BlockR[right_size].U.QuantumN.push_back(BlockR[right_size].H.QuantumN[i]);
            BlockR[right_size].U.block.push_back(rho_evec[i].transpose());
            
            tmat = BlockR[right_size].H.block[i] * rho_evec[i];
            BlockR[right_size].H.block[i] = rho_evec[i].transpose() * tmat;
            
            BlockR[right_size].H.block_size[i] = rho_evec[i].cols();
        }
        BlockR[right_size].H.ZeroPurification();
        
        
        BlockR[right_size].c_up[right_size].RhoPurification(rho);
        BlockR[right_size].c_down[right_size].RhoPurification(rho);
        
        for (int i = 0; i < n_blocks - 1; i++) {
            assert(BlockR[right_size].c_up[right_size].QuantumN[i] == BlockR[right_size].U.QuantumN[i]);
            
            BlockR[right_size].c_up[right_size].block_size[i] = rho_evec[i].cols();
            BlockR[right_size].c_down[right_size].block_size[i] = rho_evec[i].cols();
            
            if (BlockR[right_size].U.QuantumN[i] + 1 == BlockR[right_size].U.QuantumN[i + 1]) {
                
                tmat = BlockR[right_size].c_up[right_size].block[i] * rho_evec[i + 1];
                BlockR[right_size].c_up[right_size].block[i] = rho_evec[i].transpose() * tmat;
                
                tmat = BlockR[right_size].c_down[right_size].block[i] * rho_evec[i + 1];
                BlockR[right_size].c_down[right_size].block[i] = rho_evec[i].transpose() * tmat;
            } else {
                BlockR[right_size].c_up[right_size].block[i].resize(0, 0);
                BlockR[right_size].c_down[right_size].block[i].resize(0, 0);
            }
        }
        // Speical treatment to the last non-coupling matrix
        BlockR[right_size].c_up[right_size].block_size[n_blocks - 1] = rho_evec[n_blocks - 1].cols();
        BlockR[right_size].c_up[right_size].block[n_blocks - 1].resize(0, 0);
        
        BlockR[right_size].c_down[right_size].block_size[n_blocks - 1] = rho_evec[n_blocks - 1].cols();
        BlockR[right_size].c_down[right_size].block[n_blocks - 1].resize(0, 0);
        
        BlockR[right_size].c_up[right_size].ZeroPurification();
        BlockR[right_size].c_down[right_size].ZeroPurification();
    }

    return error;
}


/*
void DMRGSystem::BuildSeed(SweepDirection dir)
{
    int alpha, beta, gama;  // Real tensor indices
    MatrixXd old_psi;   // Intermediate result before tensor reshape
    MatrixXd new_psi;   // Intermediate result after tensor reshape
    
    if (dir == L2R) {
        if (left_size == 1) {
            seed = psi;
            return;
        } else {
            const MatrixXd &UL = BlockL[left_size - 1].U;
            const MatrixXd &UR = BlockR[right_size].U;
            
            old_psi = UL * psi;
            
            new_psi.resize(old_psi.rows() * d_per_site, old_psi.cols() / d_per_site);
            
            for (int i = 0; i < old_psi.rows(); i++) {
                for (int j = 0; j < old_psi.cols(); j++) {
                    alpha = i;
                    gama = j % new_psi.cols();
                    beta = (j - gama) / new_psi.cols();
                    new_psi(i * d_per_site + beta, gama) = old_psi(i, j);
                }
            }
            
            seed = new_psi * UR;
        }
    } else {
        if (right_size == 1) {
            seed = psi;
            return;
        } else {
            const MatrixXd &UL = BlockL[left_size].U;
            const MatrixXd &UR = BlockR[right_size - 1].U;
            
            old_psi = psi * UR.transpose();
            
            new_psi.resize(old_psi.rows() / d_per_site, old_psi.cols() * d_per_site);
            
            for (int i = 0; i < old_psi.rows(); i++) {
                for (int j = 0; j < old_psi.cols(); j++) {
                    gama = j;
                    beta = i % d_per_site;
                    alpha = (i - beta) / d_per_site;
                    new_psi(alpha, beta * old_psi.cols() + gama) = old_psi(i, j);
                }
            }
            
            seed = UL.transpose() * new_psi;
            
        }
    }
}
 */

/*
void DMRGSystem::Measure()
{
    const DMRGBlock &BL = BlockL[left_size];
    const DMRGBlock &BR = BlockR[right_size];
    
    // Two-point correlation function
    // According to Haldane conjecture: For spin 1/2 Heisenberg chian, it should decay algebraically.
    // For spin 1 Heisenberg chain, it should decay exponentially.
    for (int i = 0; i <= left_size; i++) {
        for (int j = 0; j <= right_size; j++) {
            cout << "Sz(" << i << ")Sz(" << nsites - j - 1 << ") = " << measure_two_site(BL.sz[i], BR.sz[j], psi) << endl;
        }
    }
     
}

double measure_local(const MatrixXd &op, const MatrixXd &psi, BlockPosition pos)
{
    double res = 0;
    MatrixXd tmat;

    if (pos == LEFT) {
        tmat = op * psi;
        res = (psi.transpose() * tmat).trace();
    } else {
        tmat = op * psi.transpose();
        res = (psi * tmat).trace();
    }
    
    return res;
}

double measure_two_site(const MatrixXd &op_left, const MatrixXd &op_right, const MatrixXd &psi)
{
    double res = 0;
    MatrixXd tmat;
    MatrixXd tmat2;
    
    tmat = op_left * psi;
    tmat2= psi.transpose() * tmat;
    res = (tmat2 * op_right).trace();
    
    return res;
}
*/
        
