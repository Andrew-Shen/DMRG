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
    sol = FailSolution::TRUNC;
    
    fermion = false;
    // Definition of the fermion order: (example)
    // c_up^dag[4] c_down^dag[4] c_up^dag[3] c_down^dag[3] |0>
    // where 0 -> 4 is left -> right
    
    left_size = 0;
    right_size = 0;
    state = SweepDirection::WR;
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

    c_up0(0,1) = 1.0;
    c_up0(2,3) = 1.0;
    c_down0(0,2) = 1.0;
    if (fermion == true) {
        // this comes from the definition of fermion order above
        c_down0(1,3) = -1.0;
    } else {
        c_down0(1,3) = 1.0;
    }
    u0(3,3) = 1.0;
    u0 = u * u0;

    sz0(1,1) = 0.5;
    sz0(2,2) = -0.5;
    n_up0(1,1) = 1.0;
    n_up0(3,3) = 1.0;
    n_down0(2,2) = 1.0;
    n_down0(3,3) = 1.0;
    
    vector<int> tsize = {0, 1, 1, 2};
    H0.UpdateQN(tsize);
    H0.UpdateBlock(u0);
    BlockL[0].H = H0;
    BlockR[0].H = BlockL[0].H;
    BlockL[0].c_up[0].UpdateQN(tsize);
    BlockL[0].c_up[0].UpdateBlock(c_up0);
    BlockR[0].c_up[0] = BlockL[0].c_up[0];
    BlockL[0].c_down[0].UpdateQN(tsize);
    BlockL[0].c_down[0].UpdateBlock(c_down0);
    BlockR[0].c_down[0] = BlockL[0].c_down[0];
    

    cout << "DMRG System initialized " << endl;
    cout << "Wavefunction transformation fail solution: ";
    if (sol == FailSolution::TRUNC) {
        cout << "Omit basis" << endl;
    } else {
        cout << "Random seed" << endl;
    }
    cout << "Particle statistics: ";
    if (fermion == true) {
        cout << "Fermion" << endl;
    } else {
        cout << "Boson" << endl;
    }
}

void DMRGSystem::WarmUp(int total_QN, int n_states_to_keep, double truncation_error)
{
    state = SweepDirection::WR;

    double density = total_QN;
    density /= nsites;
    
    for (int n = 1; n < nsites / 2; n++) {
        cout << "=== Warmup: Iteration " << n << " ===" << endl;
        
        BuildSeed((int)(2 * n + 2) * density);
        
        cout << "Block Size: Left = " << left_size << ", Right = " << right_size << endl;

        BuildBlock(BlockPosition::LEFT);
        BuildBlock(BlockPosition::RIGHT);
        
        GroundState(2 * n + 2);
        
        Truncate(BlockPosition::LEFT, n_states_to_keep, truncation_error);
        Truncate(BlockPosition::RIGHT, n_states_to_keep, truncation_error);
    }
}

void DMRGSystem::Sweep(int total_QN, int n_sweeps, int n_states_to_keep, double truncation_error)
{
    int first_iter = 0.5 * nsites;
    left_size = first_iter - 1;
    right_size = nsites - 2 - left_size;
    
    state = SweepDirection::L2R;
    for (int sweep = 0; sweep <= n_sweeps; ) {
        if (state == SweepDirection::L2R) {
            cout << "=== Sweep " << sweep << ": Left-to-right Iteration " << " ===" << endl;
        } else {
            cout << "=== Sweep " << sweep << ": Right-to-left Iteration " << " ===" << endl;
        }
        
        BuildSeed(total_QN);
        cout << "Block Size: Left = " << left_size << ", Right = " << right_size << endl;
        
        BuildBlock(BlockPosition::LEFT);
        BuildBlock(BlockPosition::RIGHT);
        
        GroundState(nsites);
        
        if (state == SweepDirection::L2R) {
            Truncate(BlockPosition::LEFT, n_states_to_keep, truncation_error);
            // if there is wavefuncation transformation, the sweep must not return until the end
            if (left_size == nsites - 3) {
                state = SweepDirection::R2L;
                sweep ++;

                left_size ++;
                right_size --;
            }
        } else {
            Truncate(BlockPosition::RIGHT, n_states_to_keep, truncation_error);
            if (right_size == nsites - 3) {
                state = SweepDirection::L2R;

                left_size --;
                right_size ++;
            }
        }
    }
    cout << "Sweep completed. " << endl;
}

void DMRGSystem::BuildSeed(int n)
{
    bool wf_transformation = false;
    
    WavefunctionBlock npsi;
    
    switch (state) {
        case SweepDirection::WR:
            left_size ++;
            right_size ++;
            break;
        case SweepDirection::L2R:
            left_size ++;
            right_size --;
            break;
        case SweepDirection::R2L:
            left_size --;
            right_size ++;
            break;
        default:
            break;
    }
    
    // build basis for left and right block
    vector<int> quantumN_left = KronQuantumN(BlockL[left_size - 1].H, H0);
    vector<int> trans_idx_left = SortIndex(quantumN_left, SortOrder::ASCENDING);
    vector<int> unsqueezed_quantumN_left = quantumN_left;
    vector<size_t> block_size_left = SqueezeQuantumN(quantumN_left);
    
    vector<int> quantumN_right = KronQuantumN(H0, BlockR[right_size - 1].H);
    vector<int> trans_idx_right = SortIndex(quantumN_right, SortOrder::ASCENDING);
    vector<int> unsqueezed_quantumN_right = quantumN_right;
    vector<size_t> block_size_right = SqueezeQuantumN(quantumN_right);
    
    if (state == SweepDirection::WR) {
        seed = InitializeWavefunction(quantumN_left, quantumN_right,
                                    block_size_left, block_size_right,
                                    n, WBType::RANDOM);
    } else {
        if ((left_size == 1 && state == SweepDirection::L2R) ||
            (right_size == 1 && state == SweepDirection::R2L)) {
            seed = psi;
            wf_transformation = false;
        } else {
            wf_transformation = true;

            npsi.quantumN_sector = n;
            npsi.block.clear();
            npsi.QuantumN.clear();
        }
    }
    
    if (wf_transformation == true) {
        // determine the quantum number and the dimension of the new wavefunction
        for (int i = 0; i < quantumN_left.size(); i++) {
            int qn_left = quantumN_left[i];
            if (qn_left > n) {
                break;
            }
            int block_idx_right = SearchIndex(quantumN_right, n - qn_left);
            if (block_idx_right == -1) {
                continue;
            }
            
            int U_idx;
            if (state == SweepDirection::L2R) {
                U_idx = BlockR[right_size].U.SearchQuantumN(n - qn_left);
            } else {
                U_idx = BlockL[left_size].U.SearchQuantumN(qn_left);
            }
            // check whether the wavefunction transformaion fails
            if (U_idx == -1) {
                if (sol == FailSolution::RAND) {
                    cout << "Wavefunction transformation failed! Will use random Lanczos seed. " << endl;
                    wf_transformation = false;
                    seed = InitializeWavefunction(quantumN_left, quantumN_right,
                                                block_size_left, block_size_right,
                                                n, WBType::RANDOM);
                    break;
                } else {
                    cout << "Wavefunction transformation failed! Will omit the lost quantum number. " << endl;
                    continue;
                }
            }
            
            npsi.QuantumN.push_back(qn_left);
            if (state == SweepDirection::L2R) {
                npsi.block.push_back(MatrixXd(block_size_left[i],
                                              BlockR[right_size].U.block[U_idx].rows()));
            } else {
                npsi.block.push_back(MatrixXd(BlockL[left_size].U.block[U_idx].rows(), block_size_right[block_idx_right]));
            }
        }
    }
    
    if (wf_transformation == true) {
        if (state == SweepDirection::L2R) {
            // A note on the notation:
            // Variables with name xxx_left or xxx_right is for Block left_size or right_size
            // Variables with name xxx_l or xxx_r is for Block left_size - 1 or right_size + 1
            // Variables with name quantumN_xxx is a vector, with name qn_xxx is an element in the vector
            vector<int> quantumN_r = KronQuantumN(H0, BlockR[right_size].H);
            vector<int> trans_idx_r = SortIndex(quantumN_r, SortOrder::ASCENDING);
            vector<size_t> block_size_r = SqueezeQuantumN(quantumN_r);
            size_t total_d_right = BlockR[right_size].H.total_d();
            
            psi.Truncate(BlockL[left_size - 1].U, BlockPosition::LEFT, false);
            
            for (int i = 0; i < psi.size(); i++) {
                int qn_l = psi.QuantumN[i];
                int block_idx_l = BlockL[left_size - 1].H.SearchQuantumN(qn_l);
                int block_idx_r = SearchIndex(quantumN_r, n - qn_l);
                
                for (int j = 0; j < psi.block[i].rows(); j++) {
                    for (int k = 0; k < psi.block[i].cols(); k++) {
                        int idx_r = trans_idx_r.at(BlockFirstIndex(block_size_r, block_idx_r) + k);
                        
                        int idx_right = idx_r % (int)total_d_right;
                        
                        double idx_site = (idx_r - idx_right) / total_d_right;
                        //assert(idx_site >= 0 && idx_site < d_per_site && floor(idx_site) == idx_site);
                        
                        int idx_left = SearchIndex(trans_idx_left, d_per_site * (BlockL[left_size - 1].H.BlockFirstIdx(block_idx_l) + j) + idx_site);
                        
                        int block_idx_left = SearchBlockIndex(block_size_left, idx_left);
                        int block_idx_right = SearchBlockIndex(BlockR[right_size].H.block_size, idx_right);
                        
                        int npsi_block = npsi.SearchQuantumN(quantumN_left[block_idx_left]);
                        int npsi_row = idx_left - BlockFirstIndex(block_size_left, block_idx_left);
                        int npsi_col = idx_right - BlockR[right_size].H.BlockFirstIdx(block_idx_right);
                        npsi.block[npsi_block](npsi_row, npsi_col) = psi.block[i](j, k);
                    }
                }
            }
            npsi.Truncate(BlockR[right_size].U, BlockPosition::RIGHT, false);
        } else {
            // A note on the notation:
            // Variables with name xxx_left or xxx_right is for Block left_size or right_size
            // Variables with name xxx_l or xxx_r is for Block left_size + 1 or right_size - 1
            // Variables with name quantumN_xxx is a vector, with name qn_xxx is an element in the vector
            vector<int> quantumN_l = KronQuantumN(BlockL[left_size].H, H0);
            vector<int> trans_idx_l = SortIndex(quantumN_l, SortOrder::ASCENDING);
            vector<size_t> block_size_l = SqueezeQuantumN(quantumN_l);
            size_t total_d_r = BlockR[right_size - 1].H.total_d();
            
            psi.Truncate(BlockR[right_size - 1].U, BlockPosition::RIGHT, true);
            
            for (int i = 0; i < psi.size(); i++) {
                int qn_l = psi.QuantumN[i];
                int block_idx_l = SearchIndex(quantumN_l, qn_l);
                int block_idx_r =  BlockR[right_size - 1].H.SearchQuantumN(n - qn_l);
                
                for (int j = 0; j < psi.block[i].rows(); j++) {
                    for (int k = 0; k < psi.block[i].cols(); k++) {
                        int idx_l = trans_idx_l[BlockFirstIndex(block_size_l, block_idx_l) + j];
                        int idx_site = idx_l % d_per_site;
                        
                        double idx_left = (idx_l - idx_site) / d_per_site;
                        //size_t total_d_left = BlockL[left_size].H.total_d();
                        //assert(idx_left >= 0 && idx_left < total_d_left && floor(idx_left) == idx_left);
                        
                        int idx_right = SearchIndex(trans_idx_right, (int)total_d_r * idx_site + BlockR[right_size - 1].H.BlockFirstIdx(block_idx_r) + k);
                        
                        int block_idx_left = SearchBlockIndex(BlockL[left_size].H.block_size, idx_left);
                        int block_idx_right = SearchBlockIndex(block_size_right, idx_right);
                        
                        int npsi_block = npsi.SearchQuantumN(BlockL[left_size].H.QuantumN[block_idx_left]);
                        int npsi_row = idx_left - BlockL[left_size].H.BlockFirstIdx(block_idx_left);
                        int npsi_col = idx_right - BlockFirstIndex(block_size_right, block_idx_right);
                        npsi.block[npsi_block](npsi_row, npsi_col) = psi.block[i](j, k);
                    }
                }
            }
            npsi.Truncate(BlockL[left_size].U, BlockPosition::LEFT, true);
        }
        // very important
        npsi.normalize();
        seed = npsi;
    }
    
    // update basis for both blocks
    BlockL[left_size].idx = trans_idx_left;
    BlockL[left_size].UpdateQN(unsqueezed_quantumN_left, left_size);
    BlockR[right_size].idx = trans_idx_right;
    BlockR[right_size].UpdateQN(unsqueezed_quantumN_right, right_size);
}

void DMRGSystem::BuildBlock(BlockPosition _position)
{
    if (_position == BlockPosition::LEFT) {
        MatrixXd HL = BlockL[left_size - 1].H.FullOperator();
        MatrixXd c_upL = BlockL[left_size - 1].c_up[left_size - 1].FullOperator();
        MatrixXd c_downL = BlockL[left_size - 1].c_down[left_size - 1].FullOperator();
        
        size_t dim_l = HL.cols();
        
        MatrixXd I_left = MatrixXd::Identity(dim_l, dim_l);
        MatrixXd I_site = MatrixXd::Identity(d_per_site, d_per_site);
        MatrixXd I_sign;
        
        if (fermion == true) {
            I_sign = H0.IdentitySign();
        } else {
            I_sign = I_site;
        }
        
        MatrixXd H = kroneckerProduct(HL, I_site) + kroneckerProduct(I_left, u0) +
        kroneckerProduct(I_left, c_up0.transpose()) * kroneckerProduct(c_upL, I_sign) +
        kroneckerProduct(c_upL.transpose(), I_sign) * kroneckerProduct(I_left, c_up0) +
        kroneckerProduct(I_left, c_down0.transpose()) * kroneckerProduct(c_downL, I_sign) +
        kroneckerProduct(c_downL.transpose(), I_sign) * kroneckerProduct(I_left, c_down0);
        
        MatrixReorder(H, BlockL[left_size].idx);
        BlockL[left_size].H.UpdateBlock(H);

        c_upL = kroneckerProduct(I_left, c_up0);
        MatrixReorder(c_upL, BlockL[left_size].idx);
        BlockL[left_size].c_up[left_size].UpdateBlock(c_upL);
        
        c_downL = kroneckerProduct(I_left, c_down0);
        MatrixReorder(c_downL, BlockL[left_size].idx);
        BlockL[left_size].c_down[left_size].UpdateBlock(c_downL);
    } else {
        MatrixXd HR = BlockR[right_size - 1].H.FullOperator();
        MatrixXd c_upR = BlockR[right_size - 1].c_up[right_size - 1].FullOperator();
        MatrixXd c_downR = BlockR[right_size - 1].c_down[right_size - 1].FullOperator();
        
        size_t dim_r = HR.cols();
        
        MatrixXd I_right = MatrixXd::Identity(dim_r, dim_r);
        MatrixXd I_site = MatrixXd::Identity(d_per_site, d_per_site);
        MatrixXd I_sign;

        if (fermion == true) {
            I_sign = BlockR[right_size - 1].H.IdentitySign();
        } else {
            I_sign = I_right;
        }

        MatrixXd H = kroneckerProduct(I_site, HR) + kroneckerProduct(u0, I_right) +
        kroneckerProduct(c_up0.transpose(), I_sign) * kroneckerProduct(I_site, c_upR) +
        kroneckerProduct(I_site, c_upR.transpose()) * kroneckerProduct(c_up0, I_sign) +
        kroneckerProduct(c_down0.transpose(), I_sign) * kroneckerProduct(I_site, c_downR) +
        kroneckerProduct(I_site, c_downR.transpose()) * kroneckerProduct(c_down0, I_sign);
        
        MatrixReorder(H, BlockR[right_size].idx);
        BlockR[right_size].H.UpdateBlock(H);
       
        c_upR = kroneckerProduct(c_up0, I_sign);
        MatrixReorder(c_upR, BlockR[right_size].idx);
        BlockR[right_size].c_up[right_size].UpdateBlock(c_upR);
        
        c_downR = kroneckerProduct(c_down0, I_sign);
        MatrixReorder(c_downR, BlockR[right_size].idx);
        BlockR[right_size].c_down[right_size].UpdateBlock(c_downR);
    }
}

void DMRGSystem::GroundState(int site)
{
    cout << "Total quantum number sector: " << seed.quantumN_sector << endl;
    cout << "Number of sites: " << site << endl;
    double ev = Lanczos(*this, max_lanczos_iter, rel_err);
    cout << "Energy per site: " << std::setprecision(12) << ev / site << endl;
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

    eig_idx = SortIndex(rho_eig_t, SortOrder::DESCENDING);
    
    error = 0;

    int _m = 0;
    double inv_error = 0;
    _max_m = min(_max_m, (int)rho_eig_t.size());
    for (int i = 0; i < _max_m; i++) {
        inv_error += rho_eig_t[i];
        if ((1 - inv_error) < _trun_err) {
            _m = i + 1;
            break;
        }
    }
    error = 1 - inv_error;
    if (_m == 0) {
        _m = _max_m;
        cout << "Max truncation number reaches. " << endl;
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
            if ((eig_idx[i] <= rho.BlockLastIdx(j)) && (eig_idx[i] >= rho.BlockFirstIdx(j))) {
                if (truncation_flag[j] != true) {
                    block_m = eig_idx[i] - rho.BlockFirstIdx(j);
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
            assert(BlockL[left_size].H.QuantumN[i] == psi.QuantumN[i]);
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
            //assert(BlockL[left_size].c_up[left_size].QuantumN[i] == BlockL[left_size].U.QuantumN[i]);
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
        
