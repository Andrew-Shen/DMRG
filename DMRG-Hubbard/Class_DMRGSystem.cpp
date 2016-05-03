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
#include <fstream>
#include <iomanip>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <complex>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;

DMRGSystem::DMRGSystem(int _nsites, int _max_lanczos_iter, double _trunc_err, double _rel_err, double u)
{
    sol = FailSolution::TRUNC;
    
    fermion = false;
    // Definition of the fermion order: (example)
    // c_up^dag[4] c_down^dag[4] c_up^dag[3] c_down^dag[3] |0>
    // where 0 -> 4 is left -> right
    
    left_size = 0;
    right_size = 0;
    state = SweepDirection::WR;
    sweep = 0;
    nsites = _nsites;
    
    BlockL.resize(nsites);
    BlockR.resize(nsites);

    max_lanczos_iter = _max_lanczos_iter;
    rel_err = _rel_err;
    trunc_err = _trunc_err;

    d_per_site = 4; // Dimension of Hilbert space per site.
    
    c_up0 = MatrixXcd::Zero(d_per_site, d_per_site);
    c_down0 = MatrixXcd::Zero(d_per_site, d_per_site);
    u0 = MatrixXcd::Zero(d_per_site, d_per_site);
    
    n_up0 = MatrixXcd::Zero(d_per_site, d_per_site);
    n_down0 = MatrixXcd::Zero(d_per_site, d_per_site);

    c_up0(0, 1) = 1.0;
    c_up0(2, 3) = 1.0;
    c_down0(0, 2) = 1.0;
    if (fermion == true) {
        // this comes from the definition of fermion order above
        c_down0(1, 3) = -1.0;
    } else {
        c_down0(1, 3) = 1.0;
    }
    u0(3, 3) = 1.0;
    u0 = u * u0;
    hubbard_u = u;

    n_up0(1, 1) = 1.0;
    n_up0(3, 3) = 1.0;
    n_down0(2, 2) = 1.0;
    n_down0(3, 3) = 1.0;
    
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
    
    time = 0;
    
    chrono::time_point<chrono::system_clock> StartTime = chrono::system_clock::now();
    time_t start_t = chrono::system_clock::to_time_t(StartTime);
    char mbstr[100];
    strftime(mbstr, sizeof(mbstr), "%c", localtime(&start_t));
    cout << "DMRG System initialized at " << mbstr << endl;
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
    
    strftime(filename, sizeof(filename), "%F %H%M%S.txt", localtime(&start_t));
    inFile.open(filename, ios::trunc);
    inFile << "Model Parameters: " << endl;
    if (fermion == true) {
        inFile << "Fermion, ";
    } else {
        inFile << "Boson, ";
    }
    
    inFile << "t = 1.0, U = " << hubbard_u << endl;
    inFile << "Site = " << nsites << ", Particles = " << psi.quantumN_sector << endl;
    inFile << "DMRG Parameters: " << endl;
    inFile << "Truncation Err = " << trunc_err << ", Lancozs Err = " << rel_err << ", ";
    if (sol == FailSolution::TRUNC) {
        inFile << "Omit basis" << endl;
    } else {
        inFile << "Random seed" << endl;
    }
    inFile << endl;
    inFile.close();

}

void DMRGSystem::WarmUp(int total_QN, int n_states_to_keep, double truncation_error)
{
    state = SweepDirection::WR;

    double density = total_QN;
    density /= nsites;
    
    if (nsites % 2) {
        cout << "Number of sites should be even! " << endl;
        abort();
    }
    
    for (int n = 1; n < 0.5 * nsites; n++) {
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

void DMRGSystem::Sweep(int total_QN, int n_sweeps, int n_states_to_keep)
{
    int first_iter = 0.5 * nsites;
    left_size = first_iter - 1;
    right_size = nsites - 2 - left_size;

    state = SweepDirection::L2R;
    while (sweep <= n_sweeps) {
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
            if (left_size == right_size && sweep > 0) {
                Measure(false);
            }
            
            Truncate(BlockPosition::LEFT, n_states_to_keep, trunc_err);
            // if there is wavefuncation transformation, the sweep must not return until the end
            if (left_size == nsites - 3) {
                state = SweepDirection::R2L;
                sweep ++;
                
                left_size ++;
                right_size --;
            }
        } else {
            Truncate(BlockPosition::RIGHT, n_states_to_keep, trunc_err);
            
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
                                    n, WBType::ONES);
        // why Lancozs fail for RAND seed for complex code? because of Lanczos
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
                npsi.block.push_back(MatrixXcd(block_size_left[i],
                                              BlockR[right_size].U.block[U_idx].rows()));
            } else {
                npsi.block.push_back(MatrixXcd(BlockL[left_size].U.block[U_idx].rows(), block_size_right[block_idx_right]));
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
        MatrixXcd HL = BlockL[left_size - 1].H.FullOperator();
        MatrixXcd c_upL = BlockL[left_size - 1].c_up[left_size - 1].FullOperator();
        MatrixXcd c_downL = BlockL[left_size - 1].c_down[left_size - 1].FullOperator();
        
        size_t dim_l = HL.cols();
        
        MatrixXcd I_left = MatrixXcd::Identity(dim_l, dim_l);
        MatrixXcd I_site = MatrixXcd::Identity(d_per_site, d_per_site);
        MatrixXcd I_sign;
        
        if (fermion == true) {
            I_sign = H0.IdentitySign();
        } else {
            I_sign = I_site;
        }
        
        MatrixXcd H = kroneckerProduct(HL, I_site) + kroneckerProduct(I_left, u0) +
        kroneckerProduct(I_left, c_up0.adjoint()) * kroneckerProduct(c_upL, I_sign) +
        kroneckerProduct(c_upL.adjoint(), I_sign) * kroneckerProduct(I_left, c_up0) +
        kroneckerProduct(I_left, c_down0.adjoint()) * kroneckerProduct(c_downL, I_sign) +
        kroneckerProduct(c_downL.adjoint(), I_sign) * kroneckerProduct(I_left, c_down0);
        
        MatrixReorder(H, BlockL[left_size].idx);
        BlockL[left_size].H.UpdateBlock(H);

        c_upL = kroneckerProduct(I_left, c_up0);
        MatrixReorder(c_upL, BlockL[left_size].idx);
        BlockL[left_size].c_up[left_size].UpdateBlock(c_upL);
        
        c_downL = kroneckerProduct(I_left, c_down0);
        MatrixReorder(c_downL, BlockL[left_size].idx);
        BlockL[left_size].c_down[left_size].UpdateBlock(c_downL);
    } else {
        MatrixXcd HR = BlockR[right_size - 1].H.FullOperator();
        MatrixXcd c_upR = BlockR[right_size - 1].c_up[right_size - 1].FullOperator();
        MatrixXcd c_downR = BlockR[right_size - 1].c_down[right_size - 1].FullOperator();
        
        size_t dim_r = HR.cols();
        
        MatrixXcd I_right = MatrixXcd::Identity(dim_r, dim_r);
        MatrixXcd I_site = MatrixXcd::Identity(d_per_site, d_per_site);
        MatrixXcd I_sign;

        if (fermion == true) {
            I_sign = BlockR[right_size - 1].H.IdentitySign();
        } else {
            I_sign = I_right;
        }

        MatrixXcd H = kroneckerProduct(I_site, HR) + kroneckerProduct(u0, I_right) +
        kroneckerProduct(c_up0.adjoint(), I_sign) * kroneckerProduct(I_site, c_upR) +
        kroneckerProduct(I_site, c_upR.adjoint()) * kroneckerProduct(c_up0, I_sign) +
        kroneckerProduct(c_down0.adjoint(), I_sign) * kroneckerProduct(I_site, c_downR) +
        kroneckerProduct(I_site, c_downR.adjoint()) * kroneckerProduct(c_down0, I_sign);
        
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

double DMRGSystem::Truncate(BlockPosition _position, int _max_m, double _trunc_err)
{
    int n = psi.quantumN_sector;
    size_t n_blocks = psi.QuantumN.size();
    
    rho.QuantumN.clear();
    rho.block.clear();
    rho.block_size.clear();
    if (_position == BlockPosition::LEFT) {
        for (int i = 0; i < n_blocks; i++) {
            rho.QuantumN.push_back(psi.QuantumN[i]);
            rho.block.push_back(psi.block[i] * psi.block[i].adjoint());
            rho.block_size.push_back(rho.block[i].cols());
        }
    } else {
        for (int i = 0; i < n_blocks; i++) {
            // Notice the i index is denoted as the quantum number of the LEFT block.
            rho.QuantumN.push_back(n - psi.QuantumN[i]);
            rho.block.push_back(psi.block[i].adjoint() * psi.block[i]);
            rho.block_size.push_back(rho.block[i].cols());
        }
        // Thus for manipulations of RIGHT block the index should be reversed.
        reverse(rho.QuantumN.begin(), rho.QuantumN.end());
        reverse(rho.block.begin(), rho.block.end());
        reverse(rho.block_size.begin(), rho.block_size.end());
    }
    
    size_t total_d = rho.total_d();
    
    vector<MatrixXcd> rho_evec(n_blocks);
    vector<double> rho_eig_t, tvec;
    vector<int> eig_idx;
    
    SelfAdjointEigenSolver<MatrixXcd> rsolver;
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
        if ((1 - inv_error) < _trunc_err) {
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
    
    MatrixXcd tmat;
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
        BlockL[left_size].U.block_size.clear();
        for (int i = 0; i < n_blocks; i++) {
            BlockL[left_size].U.QuantumN.push_back(BlockL[left_size].H.QuantumN[i]);
            BlockL[left_size].U.block.push_back(rho_evec[i].adjoint());
            BlockL[left_size].U.block_size.push_back(rho_evec[i].cols());
        }
        
        BlockL[left_size].H.Truncate(BlockL[left_size].U);
     
        BlockL[left_size].c_up[left_size].Truncate(BlockL[left_size].U);
        BlockL[left_size].c_down[left_size].Truncate(BlockL[left_size].U);
        
    } else {
        BlockR[right_size].H.RhoPurification(rho);

        BlockR[right_size].U.block.clear();
        BlockR[right_size].U.QuantumN.clear();
        BlockR[right_size].U.block_size.clear();
        for (int i = 0; i < n_blocks; i++) {
            BlockR[right_size].U.QuantumN.push_back(BlockR[right_size].H.QuantumN[i]);
            BlockR[right_size].U.block.push_back(rho_evec[i].adjoint());
            BlockR[right_size].U.block_size.push_back(rho_evec[i].cols());
        }
        
        BlockR[right_size].H.Truncate(BlockR[right_size].U);
        
        BlockR[right_size].c_up[right_size].Truncate(BlockR[right_size].U);
        BlockR[right_size].c_down[right_size].Truncate(BlockR[right_size].U);
    }

    return error;
}

void DMRGSystem::Measure(bool print_res)
{
    inFile.open(filename, ios::app);
    if (time > 0) {
        inFile << "Measurement at Time t = " << time << endl;
    } else {
        inFile << "Measurement at Sweep n = " << sweep << endl;
    }
    
    
    cout << "=== Measurement: Building Operators ===" << endl;
    vector<OperatorBlock> n_upL(left_size + 1);
    vector<OperatorBlock> n_downL(left_size + 1);
    vector<OperatorBlock> n_upR(right_size + 1);
    vector<OperatorBlock> n_downR(right_size + 1);

    for (int i = 0; i <= left_size; i++) {
        n_upL[i] = BuildDiagOperator(n_up0, i, 0, BlockPosition::LEFT);
        n_downL[i] = BuildDiagOperator(n_down0, i, 0, BlockPosition::LEFT);
    }
    for (int i = 0; i <= right_size; i++) {
        n_upR[i] = BuildDiagOperator(n_up0, i, 0, BlockPosition::RIGHT);
        n_downR[i] = BuildDiagOperator(n_down0, i, 0, BlockPosition::RIGHT);
    }
    
    cout << "=== Measurement: Local ===" << endl;
    inFile << "Local Measurement: site_i, <n(i)>, <Sz(i)>" << endl;
    for (int i = 0; i <= left_size; i++) {
        double n_up = MeasureLocalDiag(n_upL[i], psi, BlockPosition::LEFT);
        double n_down = MeasureLocalDiag(n_downL[i], psi, BlockPosition::LEFT);
        if (print_res == true) {
            cout << "n(" << i << ") = " << n_up + n_down << ", " << "Sz(" << i << ") = " << (n_up - n_down) * 0.5 << endl;
        }
        
        inFile << i << "\t" << n_up + n_down << "\t" << (n_up - n_down) * 0.5 << endl;
    }
    for (int i = right_size; i >= 0; i--) {
        double n_up = MeasureLocalDiag(n_upR[i], psi, BlockPosition::RIGHT);
        double n_down = MeasureLocalDiag(n_downR[i], psi, BlockPosition::RIGHT);
        if (print_res == true) {
            cout << "n(" << nsites - i - 1 << ") = " << n_up + n_down << ", " << "Sz(" << nsites - i - 1 << ") = " << (n_up - n_down) * 0.5 << endl;
        }
        inFile << nsites - i - 1 << "\t" << n_up + n_down << "\t" << (n_up - n_down) * 0.5 << endl;
    }
    inFile << endl;
    
    /*
    cout << "=== Measurement: Sz Correlation ===" << endl;
    inFile << "Correlation Measurement: site_i, site_j, <Sz(i)Sz(j)>" << endl;
    cout << "----- Inter-block Measurement -----" << endl;
    for (int i = 0; i <= left_size; i++) {
        for (int j = right_size; j >= 0; j--) {
            double sz = MeasureTwoDiag(n_upL[i], n_upR[j], psi) + MeasureTwoDiag(n_downL[i], n_downR[j], psi) -
            MeasureTwoDiag(n_upL[i], n_downR[j], psi) - MeasureTwoDiag(n_downL[i], n_upR[j], psi);
            sz *= 0.25;
            if (print_res == true) {
                cout << "Sz(" << i << "," << nsites - j - 1 << ") = " << sz << endl;
            }
            inFile << i << "\t" << nsites - j - 1 << "\t" << sz << endl;
        }
    }
    cout << "----- Intra-block Measurement -----" << endl;
    MatrixXcd sz0 = 0.5 * (n_up0 - n_down0);
    for (int i = 0; i < left_size; i++) {
        for (int j = i + 1; j <= left_size; j++) {
            OperatorBlock opt = BuildDiagOperator(sz0, i, j, BlockPosition::LEFT);
            double sz = MeasureLocalDiag(opt, psi, BlockPosition::LEFT);
            if (print_res == true) {
                cout << "Sz(" << i << "," <<  j << ") = " << sz << endl;
            }
            inFile << i << "\t" << j << "\t" << sz << endl;
        }
    }
    for (int i = right_size; i >= 0; i--) {
        for (int j = right_size; j > i; j--) {
            OperatorBlock opt = BuildDiagOperator(sz0, i, j, BlockPosition::RIGHT);
            double sz = MeasureLocalDiag(opt, psi, BlockPosition::RIGHT);
            if (print_res == true) {
                cout << "Sz(" << nsites - i - 1 << "," <<  nsites - j - 1 << ") = " << sz << endl;
            }
            inFile << nsites - i - 1 << "\t" << nsites - j - 1 << "\t" << sz << endl;
        }
    }
    inFile << endl;
     */
    /*
    cout << "=== Measurement: S+S- Correlation ===" << endl;
    cout << "----- Inter-block Measurement -----" << endl;
    for (int i = 0; i <= left_size; i++) {
        SuperBlock splus_l = BuildLocalOperator_splus(BlockPosition::LEFT, i);
        for (int j = right_size; j >= 0; j--) {
     
            cout << "S+(" << i << ")S-(" << nsites - j - 1 << ") = " << sz << endl;
        }
    }
     */
    
    inFile.close();
}



/*
void DMRGSystem::BuildOperator_c(BlockPosition pos, int site)
{
    MatrixXcd tmat;
    
    MatrixXcd I_site = MatrixXcd::Identity(d_per_site, d_per_site);
    if (pos == BlockPosition::LEFT) {
        // No need for site = left_size because it is built when building blocks
        //assert(site < left_size);
        for (int i = site; i < left_size; i++) {
            vector<int> quantumN_left = KronQuantumN(BlockL[i].H, H0);
            vector<int> trans_idx_left = SortIndex(quantumN_left, SortOrder::ASCENDING);
            
            tmat = kroneckerProduct(BlockL[i].c_up[site].FullOperator(), I_site);
            MatrixReorder(tmat, BlockL[i + 1].idx);
            BlockL[i + 1].c_up[site].UpdateQN(quantumN_left);
            BlockL[i + 1].c_up[site].UpdateBlock(tmat);
            if (i != left_size - 1) {
                BlockL[i + 1].c_up[site].Truncate(BlockL[i + 1].U);
            }
            
            tmat = kroneckerProduct(BlockL[i].c_down[site].FullOperator(), I_site);
            MatrixReorder(tmat, BlockL[i + 1].idx);
            BlockL[i + 1].c_down[site].UpdateQN(quantumN_left);
            BlockL[i + 1].c_down[site].UpdateBlock(tmat);
            if (i != left_size - 1) {
                BlockL[i + 1].c_down[site].Truncate(BlockL[i + 1].U);
            }
        }
    } else {
        // No need for site = right_size because it is built when building blocks
        //assert(site < right_size);
        for (int i = site; i < right_size; i++) {
            vector<int> quantumN_right = KronQuantumN(H0, BlockR[i].H);
            vector<int> trans_idx_right = SortIndex(quantumN_right, SortOrder::ASCENDING);
            
            tmat = kroneckerProduct(I_site, BlockR[i].c_up[site].FullOperator());
            MatrixReorder(tmat, BlockR[i + 1].idx);
            BlockR[i + 1].c_up[site].UpdateQN(quantumN_right);
            BlockR[i + 1].c_up[site].UpdateBlock(tmat);
            if (i != right_size - 1) {
                BlockR[i + 1].c_up[site].Truncate(BlockR[i + 1].U);
            }
            
            tmat = kroneckerProduct(I_site, BlockR[i].c_down[site].FullOperator());
            MatrixReorder(tmat, BlockR[i + 1].idx);
            BlockR[i + 1].c_down[site].UpdateQN(quantumN_right);
            BlockR[i + 1].c_down[site].UpdateBlock(tmat);
            if (i != right_size - 1) {
                BlockR[i + 1].c_down[site].Truncate(BlockR[i + 1].U);
            }
        }
    }
}
*/

OperatorBlock DMRGSystem::BuildDiagOperator(const MatrixXcd& op0, int site1, int site2, BlockPosition pos)
{
    assert(site1 < site2 || site2 == 0);
    MatrixXcd tmat;
    vector<OperatorBlock> opt;
    
    MatrixXcd I_site = MatrixXcd::Identity(d_per_site, d_per_site);
    if (pos == BlockPosition::LEFT) {
        opt.resize(left_size + 1);
        for (int i = site1; i <= left_size; i++) {
            if (i == 0) {
                vector<int> tvec = {0, 1, 1, 2};
                opt[0].UpdateQN(tvec);
                opt[0].UpdateBlock(op0);
                continue;
            }
            
            vector<int> quantumN_left = KronQuantumN(BlockL[i - 1].H, H0);
            vector<int> trans_idx_left = SortIndex(quantumN_left, SortOrder::ASCENDING);
            
            if (i == site1) {
                size_t dim_l = BlockL[i - 1].H.total_d();
                MatrixXcd I_left = MatrixXcd::Identity(dim_l, dim_l);
                tmat = kroneckerProduct(I_left, op0);
            } else {
                if (i == site2 && site2 != 0) {
                    tmat = kroneckerProduct(opt[i - 1].FullOperator(), op0);
                } else {
                    tmat = kroneckerProduct(opt[i - 1].FullOperator(), I_site);
                }
                
            }
            MatrixReorder(tmat, BlockL[i].idx);
            opt[i].UpdateQN(quantumN_left);
            opt[i].UpdateBlock(tmat);
            
            if (i != left_size) {
                opt[i].RhoPurification(BlockL[i].U);
                opt[i].Truncate(BlockL[i].U);
            }
        }
        return opt[left_size];
    } else {
        opt.resize(right_size + 1);
        for (int i = site1; i <= right_size; i++) {
            if (i == 0) {
                vector<int> tvec = {0, 1, 1, 2};
                opt[0].UpdateQN(tvec);
                opt[0].UpdateBlock(op0);
                continue;
            }
            
            vector<int> quantumN_right = KronQuantumN(H0, BlockR[i - 1].H);
            vector<int> trans_idx_right = SortIndex(quantumN_right, SortOrder::ASCENDING);
            
            if (i == site1) {
                size_t dim_r = BlockR[i - 1].H.total_d();
                MatrixXcd I_right = MatrixXcd::Identity(dim_r, dim_r);
                tmat = kroneckerProduct(op0, I_right);
            } else {
                if (i == site2 && site2 != 0) {
                    tmat = kroneckerProduct(op0, opt[i - 1].FullOperator());
                } else {
                    tmat = kroneckerProduct(I_site, opt[i - 1].FullOperator());
                }
                
            }
            MatrixReorder(tmat, BlockR[i].idx);
            opt[i].UpdateQN(quantumN_right);
            opt[i].UpdateBlock(tmat);
            
            if (i != right_size) {
                opt[i].RhoPurification(BlockR[i].U);
                opt[i].Truncate(BlockR[i].U);
            }
        }
        return opt[right_size];
    }
    
    assert("Something must be wrong! ");
    return opt[0];
}

double MeasureLocalDiag(const OperatorBlock& op, const WavefunctionBlock& psi, BlockPosition pos)
{
    double res = 0;
    MatrixXcd tmat;
    if (pos == BlockPosition::LEFT) {
        for (int i = 0; i < psi.size(); i++) {
            int qn = psi.QuantumN[i];
            int idx = op.SearchQuantumN(qn);
            if (idx == -1) {
                continue;
            }
            tmat = op.block[idx] * psi.block[i];
            res += (psi.block[i].adjoint() * tmat).trace().real();
        }
    } else {
        for (int i = 0; i < psi.size(); i++) {
            int qn = psi.QuantumN[i];
            int idx = op.SearchQuantumN(psi.quantumN_sector - qn);
            if (idx == -1) {
                continue;
            }
            tmat = op.block[idx] * psi.block[i].adjoint();
            res += (psi.block[i] * tmat).trace().real();
        }
    }
    
    return res;
}

double MeasureTwoDiag(const OperatorBlock& op_left, const OperatorBlock& op_right, const WavefunctionBlock& psi)
{
    double res = 0;
    MatrixXcd tmat, tmat2;
    
    for (int i = 0; i < psi.size(); i++) {
        int qn = psi.QuantumN[i];
        int idx_l = op_left.SearchQuantumN(qn);
        int idx_r = op_right.SearchQuantumN(psi.quantumN_sector - qn);
        if (idx_l == -1 || idx_r == -1) {
            continue;
        }
        tmat = op_left.block[idx_l] * psi.block[i];
        tmat2 = psi.block[i].adjoint() * tmat;
        res += (tmat2 * op_right.block[idx_r]).trace().real();
    }
    
    return res;
}

double MeasureTwoSuperDiag(const OperatorBlock& op_left, const OperatorBlock& op_right, const WavefunctionBlock& psi)
{
    double res = 0;
    MatrixXcd tmat, tmat2;
    
    for (int i = 0; i < psi.size(); i++) {
        int qn = psi.QuantumN[i];
        int idx_l = op_left.SearchQuantumN(qn);
        int idx_r = op_right.SearchQuantumN(psi.quantumN_sector - qn - 1);
        int idx_b = psi.SearchQuantumN(qn + 1);
        if (idx_l == -1 || idx_r == -1 || idx_b == -1) {
            continue;
        }
        tmat = op_left.block[idx_l].adjoint() * psi.block[i];
        tmat2 = psi.block[idx_b].adjoint() * tmat;
        res += (tmat2 * op_right.block[idx_r]).trace().real();
    }
    
    return res;
}

void DMRGSystem::TimeRevolution(int total_QN, double t_max, double t_step, int n_states_to_keep)
{
    Ueven0 = BondExp(0.5 * t_step);
    Uodd0 = BondExp(t_step);
    time += t_step;
    
    while (time < t_max) {
        TimeSweep(total_QN, n_states_to_keep, tSweepDirection::EVEN);
        TimeSweep(total_QN, n_states_to_keep, tSweepDirection::ODD);
        TimeSweep(total_QN, n_states_to_keep, tSweepDirection::EVEN);
        TimeSweep(total_QN, n_states_to_keep, tSweepDirection::MEASURE);
        
        time += t_step;
    }
}

void DMRGSystem::TimeSweep(int total_QN, int n_states_to_keep, tSweepDirection tdir)
{
    if (tdir == tSweepDirection::EVEN) {
        state = SweepDirection::R2L;
        right_size = 0;
        left_size = nsites - 2 - right_size;
    } else {
        state = SweepDirection::L2R;
        left_size = 0;
        right_size = nsites - 2 - left_size;
    }
    
    while (1) {
        BuildSeed(total_QN);
        cout << "=== Time Revolution at t = " << time << " ===" << endl;
        cout << "Block Size: Left = " << left_size << ", Right = " << right_size;
        //cout << seed.norm() << endl;
        BuildBlock(BlockPosition::LEFT);
        BuildBlock(BlockPosition::RIGHT);
        
        assert(nsites % 2 == 0);
        
        if (tdir == tSweepDirection::EVEN) {
            if (right_size == 1) {
                cout << ". Bond Index: " << nsites - 2 << endl;
                TimeStep(nsites - 2, tdir);
            } else {
                if (left_size == 1) {
                    cout << ". Bond Index: " << 0 << endl;
                    TimeStep(0, tdir);
                } else {
                    cout << ". Bond Index: " << left_size << endl;
                    TimeStep(left_size, tdir);
                }
            }
        } else {
            cout << ". Bond Index: " << left_size << endl;
            TimeStep(left_size, tdir);
        }
        
        if (state == SweepDirection::L2R) {
            Truncate(BlockPosition::LEFT, n_states_to_keep, trunc_err);
            
            if (left_size == right_size && tdir == tSweepDirection::MEASURE) {
                Measure(true);
            }
            
            if (left_size == nsites - 3) {
                break;
            }
        } else {
            Truncate(BlockPosition::RIGHT, n_states_to_keep, trunc_err);
            
            if (right_size == nsites - 3) {
                break;
            }
        }
    }
}

MatrixXcd DMRGSystem::BondExp(double tstep)
{
    MatrixXcd I_site = MatrixXcd::Identity(d_per_site, d_per_site);
    
    //MatrixXcd H = kroneckerProduct(I_site, I_site);
    MatrixXcd H = -1i * tstep * (kroneckerProduct(c_up0.adjoint(), c_up0) + kroneckerProduct(c_down0.adjoint(), c_down0) + kroneckerProduct(c_up0, c_up0.adjoint()) + kroneckerProduct(c_down0, c_down0.adjoint()) + kroneckerProduct(I_site, u0) + kroneckerProduct(u0, I_site));
    
    return H.exp();
}

MatrixXcd DMRGSystem::BondU(int bond_idx, tSweepDirection tdir)
{
    switch (tdir) {
        case tSweepDirection::EVEN:
            if (bond_idx % 2) {
                return MatrixXcd::Identity(d_per_site * d_per_site, d_per_site * d_per_site);
            } else {
                return Ueven0;
            }
            break;
            
        case tSweepDirection::ODD:
            if (bond_idx % 2) {
                return Uodd0;
            } else {
                return MatrixXcd::Identity(d_per_site * d_per_site, d_per_site * d_per_site);
            }
            
        case tSweepDirection::MEASURE:
            return MatrixXcd::Identity(d_per_site * d_per_site, d_per_site * d_per_site);
            
        default:
            break;
    }
}

void DMRGSystem::TimeStep(int bond_idx, tSweepDirection tdir)
{
    psi = InitializeWavefunction(seed, WBType::ZERO);
    
    if (bond_idx != 0 && bond_idx != nsites - 2) {
        // A note on the notation:
        // Variables with name xxx_left or xxx_right is for Block left_size or right_size
        // Variables with name xxx_l or xxx_r is for Block left_size - 1 or right_size - 1
        // Variables with name quantumN_xxx is a vector, with name qn_xxx is an element in the vector
        assert(bond_idx == left_size);
        
        size_t total_d_l = BlockL[left_size - 1].H.total_d();
        size_t total_d_r = BlockR[right_size - 1].H.total_d();
        MatrixXcd tpsi = MatrixXcd::Zero(d_per_site * d_per_site, total_d_l * total_d_r);

        
        //int flag1 = 0;
        //int flag2 = 0;
        for (int i = 0; i < seed.size(); i++) {
            int qn_left = seed.QuantumN[i];
            int block_idx_left = BlockL[left_size].H.SearchQuantumN(qn_left);
            int block_idx_right = BlockR[right_size].H.SearchQuantumN(seed.quantumN_sector - qn_left);
            
            for (int j = 0; j < seed.block[i].rows(); j++) {
                for (int k = 0; k < seed.block[i].cols(); k++) {
                    //cout << j << "," << k << endl;
                    int idx_left = BlockL[left_size].idx.at(BlockFirstIndex(BlockL[left_size].H.block_size, block_idx_left) + j);
                    int idx_right = BlockR[right_size].idx.at(BlockFirstIndex(BlockR[right_size].H.block_size, block_idx_right) + k);
                    
                    int idx_site_left = idx_left % d_per_site;
                    double idx_l = (idx_left - idx_site_left) / d_per_site;
                    assert(idx_l >= 0 && idx_l < total_d_l && floor(idx_l) == idx_l);
                    int idx_r = idx_right % total_d_r;
                    double idx_site_right = (idx_right - idx_r) / total_d_r;
                    assert(idx_site_right >= 0 && idx_site_right < d_per_site && floor(idx_site_right) == idx_site_right);
                    
                    int tpsi_row = d_per_site * idx_site_left + idx_site_right;
                    assert(tpsi_row >= 0 && tpsi_row < d_per_site * d_per_site);
                    int tpsi_col = total_d_r * idx_l + idx_r;
                    assert(tpsi_col >= 0 && tpsi_col < total_d_l * total_d_r);
                    
                    //flag1++;
                    tpsi(tpsi_row, tpsi_col) = seed.block[i](j, k);
                }
            }
        }
        
        MatrixXcd tmat = BondU(bond_idx, tdir) * tpsi;
        
        for (int i = 0; i < tmat.rows(); i++) {
            for (int j = 0; j < tmat.cols(); j++) {
                int idx_site_right = i % d_per_site;
                double idx_site_left = (i - idx_site_right) / d_per_site;
                assert(idx_site_left >= 0 && idx_site_left < d_per_site && floor(idx_site_left) == idx_site_left);
                int idx_r = j % total_d_r;
                double idx_l = (j - idx_r) / total_d_r;
                assert(idx_l >= 0 && idx_l < total_d_l && floor(idx_l) == idx_l);
                
                int idx_left = SearchIndex(BlockL[left_size].idx, d_per_site * idx_l + idx_site_left);
                int idx_right = SearchIndex(BlockR[right_size].idx, (int)total_d_r * idx_site_right + idx_r);
                
                int block_idx_left = SearchBlockIndex(BlockL[left_size].H.block_size, idx_left);
                int block_idx_right = SearchBlockIndex(BlockR[right_size].H.block_size, idx_right);

                int psi_idx = psi.SearchQuantumN(BlockL[left_size].H.QuantumN[block_idx_left]);
                if (psi_idx == -1 || (BlockL[left_size].H.QuantumN[block_idx_left] + BlockR[right_size].H.QuantumN[block_idx_right] != psi.quantumN_sector)) {
                    continue;
                }
                
                int psi_row = idx_left - BlockL[left_size].H.BlockFirstIdx(block_idx_left);
                int psi_col = idx_right - BlockR[right_size].H.BlockFirstIdx(block_idx_right);
                
                //flag2++;
                psi.block[psi_idx](psi_row, psi_col) = tmat(i, j);
            }
        }
        
        //assert(flag1 == flag2);
    } else {
        OperatorBlock topt;
        
        vector<int> quantumN = KronQuantumN(H0, H0);
        vector<int> trans_idx = SortIndex(quantumN, SortOrder::ASCENDING);
        
        topt.UpdateQN(quantumN);
        MatrixXcd tmat = BondU(bond_idx, tdir);

        MatrixReorder(tmat, trans_idx);
        topt.UpdateBlock(tmat);
        
        if (bond_idx == 0) {
            for (int i = 0; i < psi.size(); i++) {
                int topt_idx = topt.SearchQuantumN(psi.QuantumN[i]);
                assert(topt_idx != -1);
                psi.block[i] = topt.block[topt_idx] * seed.block[i];
            }
        } else {
            for (int i = 0; i < psi.size(); i++) {
                int topt_idx = topt.SearchQuantumN(psi.quantumN_sector - psi.QuantumN[i]);
                assert(topt_idx != -1);
                psi.block[i] =  seed.block[i] * topt.block[topt_idx];
            }
        }
    }
    
    psi.normalize();
}