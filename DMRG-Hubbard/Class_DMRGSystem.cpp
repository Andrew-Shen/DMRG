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
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/KroneckerProduct>

using namespace Eigen;
using namespace std;


DMRGSystem::DMRGSystem(int _nsites, int _max_lanczos_iter, double _rel_err)
{
    nsites = _nsites;
    
    BlockL.resize(nsites);
    BlockR.resize(nsites);

    max_lanczos_iter = _max_lanczos_iter;
    rel_err = _rel_err;

    d_per_site = 4; // Dimension of Hilbert space per site.

    //c_up0.resize(d_per_site, d_per_site);
    //c_down0.resize(d_per_site, d_per_site);
    //sz0.resize(d_per_site, d_per_site);
    //n_up0.resize(d_per_site, d_per_site);
    //n_down0.resize(d_per_site, d_per_site);
    //u0.resize(d_per_site, d_per_site);
    
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
    u = 1;
    u0 = u * u0;

    sz0(1,1) = 0.5;
    sz0(2,2) = -0.5;
    n_up0(1,1) = 1.0;
    n_up0(3,3) = 1.0;
    n_down0(2,2) = 1.0;
    n_down0(3,3) = 1.0;
    
    quantumN0 = {0, 1, 2};
    
    // is this necessary?
    BlockL[0].resize(1);
    BlockR[0].resize(1);
    
    BlockL[0].H.Update(u0, quantumN0);
    BlockR[0].H.Update(u0, quantumN0);
    
    
    BlockL[0].c_up[0].resize(2);
    BlockL[0].c_up[0].block[0].resize(1, 2);
    BlockL[0].c_up[0].block[0] << 1, 0;
    BlockL[0].c_up[0].block[1].resize(2, 1);
    BlockL[0].c_up[0].block[1] << 1, 0;
    BlockL[0].c_down[0].resize(2);
    BlockL[0].c_down[0].block[0].resize(1, 2);
    BlockL[0].c_down[0].block[0] << 0, 1;
    BlockL[0].c_down[0].block[1].resize(2, 1);
    BlockL[0].c_down[0].block[1] << 0, 1;

    BlockR[0].c_up[0].resize(2);
    BlockR[0].c_up[0].block[0].resize(1, 2);
    BlockR[0].c_up[0].block[0] << 1, 0;
    BlockR[0].c_up[0].block[1].resize(2, 1);
    BlockR[0].c_up[0].block[1] << 1, 0;
    BlockR[0].c_down[0].resize(2);
    BlockR[0].c_down[0].block[0].resize(1, 2);
    BlockR[0].c_down[0].block[0] << 0, 1;
    BlockR[0].c_down[0].block[1].resize(2, 1);
    BlockR[0].c_down[0].block[1] << 0, 1;
}

void DMRGSystem::BuildBlockLeft(int _iter)
{
    left_size = _iter;
    
    BlockL[left_size].resize(left_size + 1);
    
    MatrixXd HL = BlockL[left_size - 1].H.regenerate();
    MatrixXd c_upL = BlockL[left_size - 1].c_up[left_size - 1].super_regenerate();
    MatrixXd c_downL = BlockL[left_size - 1].c_down[left_size - 1].super_regenerate();
    
    int dim_l = (int)HL.cols();
    //cout << dim_l << " " << c_upL.cols() << endl;

    
    assert(dim_l == c_upL.cols() && "Dimensions are not properly set in the last iteration! ");
    MatrixXd I_left = MatrixXd::Identity(dim_l, dim_l);
    MatrixXd I_site = MatrixXd::Identity(d_per_site, d_per_site);

    MatrixXd H = kroneckerProduct(HL, I_site) + kroneckerProduct(I_left, u0) +
    kroneckerProduct(c_upL, c_up0.transpose()) + kroneckerProduct(c_upL.transpose(), c_up0) +
    kroneckerProduct(c_downL, c_down0.transpose()) + kroneckerProduct(c_downL.transpose(), c_down0);

    VectorXd trans_idx(dim_l * d_per_site);
    assert(H.cols() == dim_l * d_per_site && "Dimensions are not properly set in the last iteration! ");

    //cout << "number "<< left_size -1 << " : " << BlockL[left_size - 1].number.transpose() << endl;
    
    trans_idx = ordered_kron_vector(BlockL[left_size - 1].number, number0, BlockL[left_size].number);
    //cout << "number "<< left_size  << " : " << BlockL[left_size].number.transpose() << endl;

    //cout << "Reorder L" << endl;

    reorder_matrix(H, trans_idx);

    //cout << H << endl;
    BlockL[left_size].H.save_block(H, BlockL[left_size].number);
    
    c_upL = kroneckerProduct(I_left, c_up0);
    //cout << " c_up";

    reorder_matrix(c_upL, trans_idx);
    
    BlockL[left_size].c_up[left_size].save_superblock(c_upL, BlockL[left_size].number);
    //cout << "c_down"<< endl;

    c_downL = kroneckerProduct(I_left, c_down0);
    reorder_matrix(c_downL, trans_idx);
    BlockL[left_size].c_down[left_size].save_superblock(c_downL, BlockL[left_size].number);
    
}


void DMRGSystem::BuildBlockRight(int _iter)
{
    right_size = _iter;
    
    BlockR[right_size].resize(right_size + 1);
    
    MatrixXd HR = BlockR[right_size - 1].H.regenerate();
    MatrixXd c_upR = BlockR[right_size - 1].c_up[right_size - 1].super_regenerate();
    MatrixXd c_downR = BlockR[right_size - 1].c_down[right_size - 1].super_regenerate();
    
    // cout << HR.cols() << c_upR.cols() << endl;
    
    int dim_r = (int)HR.cols();
    cout << dim_r << " " << c_upR.cols() << endl;
    assert(dim_r == c_upR.cols() && "Dimensions are not properly set in the last iteration! ");

    MatrixXd I_right = MatrixXd::Identity(dim_r, dim_r);
    MatrixXd I_site = MatrixXd::Identity(d_per_site, d_per_site);
    
    MatrixXd H = kroneckerProduct(I_site, HR) + kroneckerProduct(u0, I_right) +
    kroneckerProduct(c_up0, c_upR.transpose()) + kroneckerProduct(c_up0.transpose(), c_upR) +
    kroneckerProduct(c_down0, c_downR.transpose()) + kroneckerProduct(c_down0.transpose(), c_downR);
    
    VectorXd trans_idx(dim_r * d_per_site);
    
    //cout << "number "<< right_size -1 << " : " << BlockR[right_size - 1].number.transpose() << endl;
   
    trans_idx = ordered_kron_vector(number0, BlockR[right_size - 1].number, BlockR[right_size].number);
    
    //cout << "number "<< right_size  << " : " << BlockR[right_size].number.transpose() << endl;

    
    //cout << "Reorder R" << endl;
    reorder_matrix(H, trans_idx);

    BlockR[right_size].H.save_block(H, BlockR[right_size].number);
    
    c_upR = kroneckerProduct(c_up0, I_right);
    reorder_matrix(c_upR, trans_idx);
    BlockR[right_size].c_up[right_size].save_superblock(c_upR, BlockR[right_size].number);
    
    c_downR = kroneckerProduct(c_down0, I_right);
    reorder_matrix(c_downR, trans_idx);
    BlockR[right_size].c_down[right_size].save_superblock(c_downR, BlockR[right_size].number);
    
}

void DMRGSystem::GroundState(int n, bool wf_prediction)
{
    double ev;
        
    ev = Lanczos(*this, n, max_lanczos_iter, rel_err, wf_prediction);
    
    cout << "Energy per site: " << ev / nsites << endl;
}



double DMRGSystem::Truncate(BlockPosition _position, int _max_m, double _trun_err)
{
    int n = psi.total_particle_number;
    
    MatrixXd tmat;
    
    //cout << "n= " << n << endl;

    rho.resize(n);
    if (_position == BlockPosition::LEFT) {
        for (int i = 0; i <= n; i++) {
            rho.block[i] = psi.block[i] * psi.block[i].transpose();
            
            //cout << "i=" << i << rho.block[i].cols() << endl;
            //cout << rho.block[i] << endl;
        }
    } else {
        for (int i = 0; i <= n; i++) {
            // Notice the i index is denoted as the number of particles in the LEFT block.
            // Thus for manipulations of RIGHT block the index should be reversed.
            
            tmat = psi.block[i].transpose() * psi.block[i];
            
            rho.block[n - i] = tmat;
            
            //cout << "n-i=" << n - i << " "<< psi.block[i].rows() << "x" << psi.block[i].cols() << endl;
            //cout << rho.block[n - i].cols() << endl;
            //cout << rho.block[n - i] << endl;

            
        }
    }
    
    //cout << "norm of psi "<< psi.norm() << endl;
    int n_max_left = (int)BlockL[left_size].number.size() - 1;

    int nblocks = min(rho.max_particle_number, n_max_left) + 1;
    //cout << nblocks << endl;
    int _m = 0;
    //vector<VectorXd> rho_eig(nblocks);
    vector<MatrixXd> rho_evec(nblocks);
    
    int d = 0;
    VectorXd number;
    if (_position == BlockPosition::LEFT) {
        number = BlockL[left_size].number;
        BlockL[left_size].U.resize(nblocks - 1);
        BlockL[left_size].number.conservativeResize(nblocks);
    } else {
        number = BlockR[right_size].number;
        BlockR[right_size].U.resize(nblocks - 1);
        BlockR[right_size].number.conservativeResize(nblocks);

    }
    for (int i = 0; i < nblocks; i++) {
        // Always start from 0??
        d += number(i);
    }
    number.conservativeResize(nblocks);
    //cout << "Number before trunc: " << number.transpose() << endl;
    //cout << "d=" << d << endl;
    VectorXd rho_eig_t = VectorXd::Zero(d); // Very important to initialize to zero.
    VectorXd eig_idx(d);

    //double tmp = 0;
    //int tmp2 = 0;
    int indicator = 0;
    SelfAdjointEigenSolver<MatrixXd> rsolver;
    for (int i = 0; i < nblocks; i++) {
        //cout << rho.block[i] << endl;
        if (rho.block[i].size() == 0) {
            //cout << "rho zero " << i << endl;
            rho_evec[i].resize(0, 0);
            continue;
        }
        //assert(rho.block[i].size() != 0);
        rsolver.compute(-rho.block[i]);
        if (rsolver.info() != Success) abort();
        rho_eig_t.segment(indicator, rho.block[i].cols()) = -rsolver.eigenvalues();
        //cout << i;
        //cout << rho_eig_t.segment(indicator, rho.block[i].cols()).transpose() << endl;
        //tmp += rho_eig_t.segment(indicator, rho.block[i].cols()).sum();
        //cout << rho.block[i].cols() << endl;
        //tmp2 += rho.block[i].cols();
        indicator += rho.block[i].cols();
        rho_evec[i] = rsolver.eigenvectors();
        
        //cout << rho_evec[i] << endl;
    }
    //cout << "tmp " << tmp << "tmp2" << tmp2 << endl;
    //cout << "Energy Eigen: " << rho_eig_t.transpose() << endl;
    
        //cout << rho_eig_t << "t"<< endl;
    rho_eig_t = -rho_eig_t;
    eig_idx = sort_indexes(rho_eig_t);
    rho_eig_t = -rho_eig_t;
    //cout << "Energy Eigen: " << rho_eig_t.transpose() << endl;
    //cout << "Eig index: " << eig_idx.transpose() << endl;
    
    error = 0;
    //double su = rho_eig_t.sum();
    //su = rho_eig_t.sum();
    //cout << "size" << rho_eig_t.size() <<" su=" << su << endl;

    //cout << "size" << rho_eig_t.size() <<" su=" << su << endl;
    //assert(su == 1 && "Trace of density matrix is not 1! ");
    
    double inv_error = 0;
    for (int i = 0; i < _max_m; i++) {
        inv_error += rho_eig_t(i);
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
    
    int block_m;
    VectorXd tnumber = number;
    vector<bool> truncation_flag(nblocks);
    for (int i = 0; i < nblocks; i++) {
        truncation_flag[i] = false;
    }
 
    //cout << number.transpose() << endl;
    
    for (int i = _m; i < d; i++) {
        for (int j = 0; j < nblocks; j++) {
            if ((eig_idx(i) <= block_end(number, j)) && (eig_idx(i) >= block_begin(number, j))) {
                if (truncation_flag[j] != true) {
                    block_m = eig_idx(i) - block_begin(number, j);
                    if (block_m == 0 || rho_evec[j].rows() == 0) {
                        rho_evec[j].resize(0, 0);
                    } else {
                        //cout << j << endl;
                        //cout << rho_evec[j] << endl;
                        tmat = rho_evec[j].leftCols(block_m);
                        rho_evec[j] = tmat;
                        //cout << "U" << j << "modified. " << endl;
                        //cout << rho_evec[j] << endl;
                        
                    }
                    tnumber(j) = block_m;
                    
                    truncation_flag[j] = true;
                    cout << "Trunced idx: " << eig_idx(i) << " at block " << j << " remain " << tnumber(j) << endl;

                }
                break;
            }
        }
    }
    
    number = tnumber;
    tnumber.resize(0);
    
    //cout << rho_evec[0] << endl;
    //cout << rho_evec[1] << endl;
    //cout << rho_evec[2] << endl;
    MatrixXd U;
    int dh, du, dh2, du2;
    if (_position == BlockPosition::LEFT) {
        BlockL[left_size].number.conservativeResize(nblocks);
        BlockL[left_size].H.resize(nblocks - 1); // Additional -1 here is because the arg here is the max particle number
        BlockL[left_size].c_up[left_size].resize(nblocks - 2);
        BlockL[left_size].c_down[left_size].resize(nblocks - 2);
        
        for (int i = 0; i < nblocks; i++) {
            dh = (int)BlockL[left_size].H.block[i].cols();
            du = (int)rho_evec[i].rows();
            if (du == 0) {
                //cout << "Total truncation of H encountered at L " << i << endl;
                BlockL[left_size].number(i) = 0;
                BlockL[left_size].U.block[i].resize(0, 0);
                BlockL[left_size].H.block[i].resize(0, 0);
            } else {
                assert(dh == du && "Dimension of Hamiltonian and truncation do not agree! ");
                BlockL[left_size].number(i) = number(i);
                BlockL[left_size].U.block[i] = rho_evec[i].transpose();
                
                //cout << "U "<< i << endl;
                //cout << BlockL[left_size].U.block[i] << endl;
                
                tmat = BlockL[left_size].H.block[i] * rho_evec[i];
                BlockL[left_size].H.block[i] = rho_evec[i].transpose() * tmat;
            }
        }
        
        for (int i = 0; i < nblocks - 1; i++) {
            dh = (int)BlockL[left_size].c_up[left_size].block[i].rows();
            dh2 = (int)BlockL[left_size].c_up[left_size].block[i].cols();
            du = (int)rho_evec[i].rows();
            du2 = (int)rho_evec[i + 1].rows();
            if ((du == 0) || (du2 == 0)) {
                //cout << "Total truncation of c encountered at L " << i << endl;
                BlockL[left_size].c_up[left_size].block[i].resize(0, 0);
                BlockL[left_size].c_down[left_size].block[i].resize(0, 0);
            } else {
                assert(dh == du && dh2 == du2 && "Dimension of coupling operator and truncation do not agree! ");
                
                tmat = BlockL[left_size].c_up[left_size].block[i] * rho_evec[i + 1];
                BlockL[left_size].c_up[left_size].block[i] = rho_evec[i].transpose() * tmat;
                
                tmat = BlockL[left_size].c_down[left_size].block[i] * rho_evec[i + 1];
                BlockL[left_size].c_down[left_size].block[i] = rho_evec[i].transpose() * tmat;

            }
        }
    } else {
        BlockR[right_size].number.conservativeResize(nblocks);
        
        //cout << "R number " << right_size << endl;
        //cout << BlockR[right_size].number.transpose() << endl;
        
        BlockR[right_size].H.resize(nblocks - 1); // Additional -1 here is because the arg here is the max particle number
        BlockR[right_size].c_up[right_size].resize(nblocks - 2);
        BlockR[right_size].c_down[right_size].resize(nblocks - 2);

        
        for (int i = 0; i < nblocks; i++) {
            dh = (int)BlockR[right_size].H.block[i].cols();
            du = (int)rho_evec[i].rows();
            if (du == 0) {
                //cout << "Total truncation of H encountered at R " << i << endl;
                BlockR[right_size].number(i) = 0;
                BlockR[right_size].U.block[i].resize(0, 0);
                BlockR[right_size].H.block[i].resize(0, 0);
            } else {
                assert(dh == du && "Dimension of Hamiltonian and truncation do not agree! ");
                
                BlockR[right_size].number(i) = number(i);
                BlockR[right_size].U.block[i] = rho_evec[i].transpose();
                tmat = BlockR[right_size].H.block[i] * rho_evec[i];
                BlockR[right_size].H.block[i] = rho_evec[i].transpose() * tmat;
                cout << "size "<< BlockR[right_size].H.block[i].cols() << endl;
 
            }
        }
        //cout << nblocks << endl;
        for (int i = 0; i < nblocks - 1; i++) {
            //cout << "i=" << i << endl;
            //cout << "right_size = " << right_size << endl;
            dh = (int)BlockR[right_size].c_up[right_size].block[i].rows();
            //cout << "k" << dh << endl;
            dh2 = (int)BlockR[right_size].c_up[right_size].block[i].cols();
            //cout << "k" << dh2 << endl;

            du = (int)rho_evec[i].rows();
            //cout << "k" << du << endl;

            du2 = (int)rho_evec[i + 1].rows();

            if ((du == 0) || (du2 == 0)) {
                //cout << "Total truncation of c encountered at R " << i << endl;
                BlockR[right_size].c_up[right_size].block[i].resize(0, 0);
                BlockR[right_size].c_down[right_size].block[i].resize(0, 0);
            } else {
                assert(dh == du && dh2 == du2 && "Dimension of coupling operator and truncation do not agree! ");
                
                tmat = BlockR[right_size].c_up[right_size].block[i] * rho_evec[i + 1];
                BlockR[right_size].c_up[right_size].block[i] = rho_evec[i].transpose() * tmat;
                
                tmat = BlockR[right_size].c_down[right_size].block[i] * rho_evec[i + 1];
                BlockR[right_size].c_down[right_size].block[i] = rho_evec[i].transpose() * tmat;
            }
            
        }
    }

    cout << "Number after trunc: " << number.transpose() << endl;
    //cout << psi.block[0] << endl;
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
        
