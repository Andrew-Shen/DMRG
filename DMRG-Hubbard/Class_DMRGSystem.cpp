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
    vector<int> trans_idx = sort_index(quantumN, SortOrder::ASCENDING);
    
    BlockL[left_size].idx = trans_idx;
    
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
    vector<int> trans_idx = sort_index(quantumN, SortOrder::ASCENDING);
    
    BlockR[right_size].idx = trans_idx;
    
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
    
    rho.QuantumN.clear();
    rho.block.clear();
    rho.block_size.clear();
    if (_position == BlockPosition::LEFT) {
        for (int i = 0; i < n_blocks; i++) {
            rho.QuantumN.push_back(psi.QuantumN[i]);
            rho.block.push_back(psi.block[i] * psi.block[i].transpose());
            rho.block_size.push_back(rho.block[i].cols());
        }
    } else {
        for (int i = 0; i < n_blocks; i++) {
            // Notice the i index is denoted as the quantum number of the LEFT block.
            rho.QuantumN.push_back(n - psi.QuantumN[i]);
            rho.block.push_back(psi.block[i].transpose() * psi.block[i]);
            rho.block_size.push_back(rho.block[i].cols());
        }
        // Thus for manipulations of RIGHT block the index should be reversed.
        reverse(rho.QuantumN.begin(), rho.QuantumN.end());
        reverse(rho.block.begin(), rho.block.end());
        reverse(rho.block_size.begin(), rho.block_size.end());
    }
    
    size_t total_d = rho.total_d();
    
    vector<MatrixXd> rho_evec(n_blocks);
    vector<double> rho_eig_t, tvec;
    vector<int> eig_idx;
    
    SelfAdjointEigenSolver<MatrixXd> rsolver;
    for (int i = 0; i < n_blocks; i++) {
        // The minus sign is for the descending order of the eigenvalue.
        rsolver.compute(-rho.block[i]);
        if (rsolver.info() != Success) abort();
        
        tvec.resize(rsolver.eigenvalues().size());
        VectorXd::Map(&tvec[0], rsolver.eigenvalues().size()) = -rsolver.eigenvalues();
        rho_eig_t.insert(rho_eig_t.end(), tvec.begin(), tvec.end());
        
        rho_evec[i] = rsolver.eigenvectors();
    }
    
    eig_idx = sort_index(rho_eig_t, SortOrder::DESCENDING);
    
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
    
    MatrixXd tmat;
    for (int i = _m; i < total_d; i++) {
        for (int j = 0; j < n_blocks; j++) {
            if ((eig_idx[i] <= rho.end(j)) && (eig_idx[i] >= rho.begin(j))) {
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

void DMRGSystem::BuildSeed(int n, SweepDirection dir)
{
    WavefunctionBlock npsi;
    npsi.quantumN_sector = n;
    npsi.block.clear();
    npsi.QuantumN.clear();
    
    if (dir == SweepDirection::L2R) {
        left_size ++;
        right_size --;
        if (left_size == 1) {
            seed = psi;
            return;
        } else {
            
            // build basis for left and right block
            vector<int> qn_left = QuantumN_kron(BlockL[left_size - 1].H, H0);
            vector<int> trans_idx_left = sort_index(qn_left, SortOrder::ASCENDING);
            vector<size_t> size_left = SqueezeQuantumN(qn_left);
            assert(qn_left.size() == size_left.size());
            
            vector<int> qn_right = QuantumN_kron(H0, BlockR[right_size - 1].H);
            vector<int> trans_idx_right = sort_index(qn_right, SortOrder::ASCENDING);
            vector<size_t> size_right = SqueezeQuantumN(qn_right);
            assert(qn_right.size() == size_right.size());
  
            int left_qn, right_idx;
            
            for (int i = 0; i < qn_left.size(); i++) {
                left_qn = qn_left[i];
                if (left_qn > n) {
                    break;
                }
                right_idx = SearchIndex(qn_right, n - left_qn);
                if (right_idx == -1) {
                    continue;
                }
                npsi.QuantumN.push_back(left_qn);
                MatrixXd tmat = MatrixXd::Ones(size_left[i], size_right.at(right_idx));
                tmat *= 1e99;
                npsi.block.push_back(tmat);
            }
            //npsi.normalize();
        
            for (int i = 0; i < npsi.size(); i++) {
                assert(npsi.block[i].cols() == BlockR[right_size].U.block[BlockR[right_size].U.SearchQuantumN(n - npsi.QuantumN[i])].cols());
                npsi.block[i].resize(npsi.block[i].rows(), BlockR[right_size].U.block[BlockR[right_size].U.SearchQuantumN(n - npsi.QuantumN[i])].rows());
            }

            // wavefunction transformation
            psi.Truncation(BlockL[left_size - 1].U, BlockPosition::LEFT);
            
            for (int i = 0; i < psi.size(); i++) {
                int qn = psi.QuantumN[i];
                int block_left = BlockL[left_size - 1].H.SearchQuantumN(qn);
                
                for (int j = 0; j < psi.block[i].rows(); j++) {
                    for (int k = 0; k < psi.block[i].cols(); k++) {
                        //cout << "Block " << i << ", Row: " << j << ", Col: " << k << endl;
                        //cout << "left_size = " << left_size << " right_size = " << right_size << endl;
                        
                        vector<int> qn_r = QuantumN_kron(H0, BlockR[right_size].H);
                        vector<int> qn_r_idx = sort_index(qn_r, SortOrder::ASCENDING);
                        vector<size_t> size_r = SqueezeQuantumN(qn_r);
                        
                        int block_right = SearchIndex(qn_r, n - qn);
                        
                        //cout << block_right << endl;
    
                        int npsi_idx_right = BlockFirstIndex(size_r, block_right) + k;
                        
                        //cout << BlockR[right_size].H.total_d() << " " << BlockR[right_size].idx.size() << endl;
                        
                        npsi_idx_right = BlockR[right_size + 1].idx.at(npsi_idx_right);
                        
                        size_t total_d_right = BlockR[right_size].H.total_d();
                        double beta_idx = (npsi_idx_right - npsi_idx_right % (int)total_d_right) / total_d_right;
                        
                        
                        assert(beta_idx >= 0 && beta_idx < d_per_site && floor(beta_idx) == beta_idx);
                        
                        int npsi_idx_left = BlockL[left_size - 1].H.begin(block_left) + j;
                        npsi_idx_left = d_per_site * npsi_idx_left + beta_idx;
                        npsi_idx_left = SearchIndex(trans_idx_left, npsi_idx_left);
                        
                        // order is important
                        int npsi_block = SearchBlockIndex(size_left, npsi_idx_left);
                        int npsi_row = npsi_idx_left - BlockFirstIndex(size_left, npsi_block);
                        npsi_block = npsi.SearchQuantumN(qn_left.at(npsi_block));
                        assert(npsi_block != -1);
        
                        
                        int npsi_col = npsi_idx_right % (int)total_d_right;
                        assert(npsi_col >= 0 && npsi_col < total_d_right);
                        int r_block = SearchBlockIndex(BlockR[right_size].H.block_size, npsi_col);
                        assert(BlockR[right_size].U.block[BlockR[right_size].U.SearchQuantumN(n - npsi.QuantumN[npsi_block])].rows() == BlockR[right_size].H.block_size[r_block]);
                        assert(npsi.block[npsi_block].cols() == BlockR[right_size].H.block_size[r_block]);
                        npsi_col = npsi_col - BlockR[right_size].H.begin(r_block);
                        
                        npsi.block[npsi_block](npsi_row, npsi_col) = psi.block[i](j, k);
                    }
                }
            }
            npsi.Truncation(BlockR[right_size].U, BlockPosition::RIGHT);
            // this is very imporant
            npsi.normalize();
            seed = npsi;
            
        }
        
    } else {
        // right
    }
    
}


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
        
