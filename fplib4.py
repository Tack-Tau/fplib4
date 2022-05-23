import numpy as np
from scipy.optimize import linear_sum_assignment
import rcovdata
# import numba

class FingerPrint(object):
    """Fingerprint module for ASE-fp_GD_calculator interface.
    
        Implemented Properties:
        
            'energy': Sum of atomic fingerprint distance (L2 norm of two atomic 
                                                          fingerprint vectors)
            
            'forces': Gradient of fingerprint energy, using Hellmann–Feynman theorem
            
            'stress': Cauchy stress tensor using finite difference method

        Parameters:

            atoms:  object
                Attach an atoms object to the calculator.

            contract: bool
                Calculate fingerprint vector in contracted Guassian-type orbitals or not
            
            ntyp: int
                Number of different types of atoms in unit cell
            
            nx: int
                Maximum number of atoms in the sphere with cutoff radius for specific cell site
                
            lmax: int
                Integer to control whether using s orbitals only or both s and p orbitals for 
                calculating the Guassian overlap matrix (0 for s orbitals only, other integers
                will indicate that using both s and p orbitals)
                
            cutoff: float
                Cutoff radius for f_c(r) (smooth cutoff function) [amp], unit in Angstroms
                
    """
    
    def __init__(self,
                 contract = False,
                 ntyp = None,
                 nx = None,
                 lmax = None,
                 cutoff = None):
        if contract is not None:
            self.contract = contract
        if ntyp is not None:
            self.ntyp = int(ntyp)
        else:
            raise Exception("Number of different types of atoms in unit cell undefined!")
        if nx is not None:
            self.nx = int(nx)
        else:
            raise Exception("Maximum number of atoms in the sphere undefined!")
        if lmax is not None:
            self.lmax = int(lmax)
        else:
            raise Exception("Integer to control whether using s orbitals only or \
                             both s and p orbitals undefined!")
        if cutoff is not None:
            self.cutoff = float(cutoff)
        else:
            raise Exception("Cutoff radius undefined!")
    
    @staticmethod
    def readvasp(vp):
        buff = []
        with open(vp) as f:
            for line in f:
                buff.append(line.split())

        lat = np.array(buff[2:5], float) 
        try:
            typt = np.array(buff[5], int)
        except:
            del(buff[5])
            typt = np.array(buff[5], int)
        nat = sum(typt)
        pos = np.array(buff[7:7 + nat], float)
        types = []
        for i in range(len(typt)):
            types += [i+1]*typt[i]
        types = np.array(types, int)
        rxyz = np.dot(pos, lat)
        # rxyz = pos
        return lat, rxyz, types
    
    @staticmethod
    def read_types(vp):
        buff = []
        with open(vp) as f:
            for line in f:
                buff.append(line.split())
        try:
            typt = np.array(buff[5], int)
        except:
            del(buff[5])
            typt = np.array(buff[5], int)
        types = []
        for i in range(len(typt)):
            types += [i+1]*typt[i]
        types = np.array(types, int)
        return types
    
    @staticmethod
    def kron_delta(i,j):
        if i == j:
            m = 1.0
        else:
            m = 0.0
        return m
    
    @staticmethod
    def get_ixyz(lat, cutoff):
        lat2 = np.matmul(lat, np.transpose(lat))
        # print lat2
        val = np.linalg.eigvals(lat2)
        # print (vec)
        ixyz = int(np.sqrt(1.0/max(val))*cutoff) + 1
        return ixyz
    
    @staticmethod
    def get_rxyz_delta(rxyz):
        nat = len(rxyz)
        rxyz_delta = np.subtract( np.random.rand(nat, 3), 0.5*np.ones((nat, 3)) )
        for iat in range(nat):
            r_norm = np.linalg.norm(rxyz_delta[iat])
            rxyz_delta[iat] = np.divide(rxyz_delta[iat], r_norm)
        # rxyz_plus = np.add(rxyz, rxyz_delta)
        # rxyz_minus = np.subtract(rxyz, rxyz_delta)

        return rxyz_delta
    
    def get_common_sphere(self, lat, rxyz, types, znucl, iat, jat):
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        amp_j, n_sphere_j, icenter_j, rxyz_sphere_j, rcov_sphere_j = \
                    self.get_sphere(lat, rxyz, types, znucl, jat)
        i_sphere_count = 0
        nat_j_sphere = len(rxyz_sphere_j)
        iat_in_j_sphere = False
        rxyz_list = rxyz.tolist()
        rxyz_sphere_j_list = rxyz_sphere_j.tolist()
        for j in range(nat_j_sphere):
            if rxyz_list[iat] == rxyz_sphere_j_list[j]:
                iat_in_j_sphere = True
                return iat_in_j_sphere, j
            else:
                return iat_in_j_sphere, j
            
    def get_sphere(self, lat, rxyz, types, znucl, iat):
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        cutoff = self.cutoff
        if lmax == 0:
            lseg = 1
            l = 1
        else:
            lseg = 4
            l = 2
        ixyz = FingerPrint.get_ixyz(lat, cutoff)
        NC = 3
        wc = cutoff / np.sqrt(2.* NC)
        fc = 1.0 / (2.0 * NC * wc**2)
        nat = len(rxyz)
        cutoff2 = cutoff**2 
        n_sphere_list = []
        # print ("init iat = ", iat)
        # if iat > (nat-1):
            # print ("max iat = ", iat)
            # sys.exit("Error: ith atom (iat) is out of the boundary of the original unit cell \
            # (POSCAR)")
            # return amp, n_sphere, rxyz_sphere, rcov_sphere
        # else:
            # print ("else iat = ", iat)
        if iat <= (nat-1):
            rxyz_sphere = []
            rcov_sphere = []
            ind = [0] * (lseg * nx)
            amp = []
            xi, yi, zi = rxyz[iat]
            n_sphere = 0
            for jat in range(nat):
                for ix in range(-ixyz, ixyz+1):
                    for iy in range(-ixyz, ixyz+1):
                        for iz in range(-ixyz, ixyz+1):
                            xj = rxyz[jat][0] + ix*lat[0][0] + iy*lat[1][0] + iz*lat[2][0]
                            yj = rxyz[jat][1] + ix*lat[0][1] + iy*lat[1][1] + iz*lat[2][1]
                            zj = rxyz[jat][2] + ix*lat[0][2] + iy*lat[1][2] + iz*lat[2][2]
                            d2 = (xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2
                            if d2 <= cutoff2:
                                n_sphere += 1
                                if n_sphere > nx:
                                    print ("FP WARNING: the cutoff is too large.")
                                amp.append((1.0-d2*fc)**NC)
                                # print (1.0-d2*fc)**NC
                                rxyz_sphere.append([xj, yj, zj])
                                rcov_sphere.append(rcovdata.rcovdata[znucl[types[jat]-1]][2])
                                if jat == iat and ix == 0 and iy == 0 and iz == 0:
                                    ityp_sphere = 0
                                    icenter = n_sphere - 1
                                else:
                                    ityp_sphere = types[jat]
                                
                                for il in range(lseg):
                                    if il == 0:
                                        # print len(ind)
                                        # print ind
                                        # print il+lseg*(n_sphere-1)
                                        ind[il+lseg*(n_sphere-1)] = ityp_sphere * l
                                    else:
                                        ind[il+lseg*(n_sphere-1)] = ityp_sphere * l + 1
                                
            n_sphere_list.append(n_sphere)
            rxyz_sphere = np.array(rxyz_sphere, float)
        # for n_iter in range(nx-n_sphere+1):
            # rxyz_sphere.append([0.0, 0.0, 0.0])
            # rxyz_sphere.append([0.0, 0.0, 0.0])
        # rxyz_sphere = np.array(rxyz_sphere, float)
        # print ("amp", amp)
        # print ("n_sphere", n_sphere)
        # print ("rxyz_sphere", rxyz_sphere)
        # print ("rcov_sphere", rcov_sphere)
        return amp, n_sphere, icenter, rxyz_sphere, rcov_sphere
    
    def get_gom(self, rxyz, rcov, amp):
        lmax = self.lmax
        if lmax == 0:
            lseg = 1
            # l = 1
        else:
            lseg = 4
            # l = 2
        # s orbital only lseg == 1
        nat = len(rxyz)    
        if lseg == 1:
            om = np.zeros((nat, nat))
            for iat in range(nat):
                for jat in range(nat):
                    d = rxyz[iat] - rxyz[jat]
                    d2 = np.vdot(d, d)
                    r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                    om[iat][jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                        * np.exp(-1.0*d2*r) * amp[iat] * amp[jat]
        else:
            # for both s and p orbitals
            om = np.zeros((4*nat, 4*nat))
            for iat in range(nat):
                for jat in range(nat):
                    d = rxyz[iat] - rxyz[jat]
                    d2 = np.vdot(d, d)
                    r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                    om[4*iat][4*jat] = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 \
                        * np.exp(-1.0*d2*r) * amp[iat] * amp[jat]

                    # <s_i | p_j>
                    sji = np.sqrt(4.0*r*rcov[iat]*rcov[jat])**3 * np.exp(-1*d2*r)
                    stv = np.sqrt(8.0) * rcov[jat] * r * sji
                    om[4*iat][4*jat+1] = stv * d[0] * amp[iat] * amp[jat]
                    om[4*iat][4*jat+2] = stv * d[1] * amp[iat] * amp[jat]
                    om[4*iat][4*jat+3] = stv * d[2] * amp[iat] * amp[jat]

                    # <p_i | s_j> 
                    stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                    om[4*iat+1][4*jat] = stv * d[0] * amp[iat] * amp[jat]
                    om[4*iat+2][4*jat] = stv * d[1] * amp[iat] * amp[jat]
                    om[4*iat+3][4*jat] = stv * d[2] * amp[iat] * amp[jat]

                    # <p_i | p_j>
                    stv = -8.0 * rcov[iat] * rcov[jat] * r * r * sji
                    om[4*iat+1][4*jat+1] = stv * (d[0] * d[0] - 0.5/r) * amp[iat] * amp[jat]
                    om[4*iat+1][4*jat+2] = stv * (d[1] * d[0]        ) * amp[iat] * amp[jat]
                    om[4*iat+1][4*jat+3] = stv * (d[2] * d[0]        ) * amp[iat] * amp[jat]
                    om[4*iat+2][4*jat+1] = stv * (d[0] * d[1]        ) * amp[iat] * amp[jat]
                    om[4*iat+2][4*jat+2] = stv * (d[1] * d[1] - 0.5/r) * amp[iat] * amp[jat]
                    om[4*iat+2][4*jat+3] = stv * (d[2] * d[1]        ) * amp[iat] * amp[jat]
                    om[4*iat+3][4*jat+1] = stv * (d[0] * d[2]        ) * amp[iat] * amp[jat]
                    om[4*iat+3][4*jat+2] = stv * (d[1] * d[2]        ) * amp[iat] * amp[jat]
                    om[4*iat+3][4*jat+3] = stv * (d[2] * d[2] - 0.5/r) * amp[iat] * amp[jat]

        # for i in range(len(om)):
        #     for j in range(len(om)):
        #         if abs(om[i][j] - om[j][i]) > 1e-6:
        #             print ("ERROR", i, j, om[i][j], om[j][i])
        return om
    
    def get_D_gom(self, rxyz, rcov, amp, D_n, icenter):
        lmax = self.lmax
        cutoff = self.cutoff
        if lmax == 0:
            lseg = 1
            # l = 1
        else:
            lseg = 4
            # l = 2
        # s orbital only lseg == 1
        NC = 3
        wc = cutoff / np.sqrt(2.* NC)
        fc = 1.0 / (2.0 * NC * wc**2)
        nat = len(rxyz)    
        if lseg == 1:
            D_om = np.zeros((3, nat, nat))
            for x in range(3):
                for iat in range(nat):
                    for jat in range(nat):
                        d = rxyz[iat] - rxyz[jat]
                        dnc = rxyz[D_n] - rxyz[icenter]
                        d2 = np.vdot(d, d)
                        dnc2 = np.vdot(dnc, dnc)
                        r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                        sji = np.sqrt( 4.0*r*(rcov[iat]*rcov[jat]) )**3 * np.exp(-1.0*d2*r)
                        # Derivative of <s_i | s_j>
                        D_om[x][iat][jat] = \
                        - ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) ) * \
                          (2.0*r) * d[x] * sji * amp[iat] * amp[jat]                                \
                        - 2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) * sji * amp[iat] *   \
                                                                                       amp[jat] *   \
                          ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) )

        else:
            # for both s and p orbitals
            D_om = np.zeros((3, 4*nat, 4*nat))
            for x in range(3):
                for iat in range(nat):
                    for jat in range(nat):
                        d = rxyz[iat] - rxyz[jat]
                        dnc = rxyz[D_n] - rxyz[icenter]
                        d2 = np.vdot(d, d)
                        dnc2 = np.vdot(dnc, dnc)
                        r = 0.5/(rcov[iat]**2 + rcov[jat]**2)
                        sji = np.sqrt(4.0*r*rcov[iat]*rcov[jat])**3 * np.exp(-1.0*d2*r)
                        # Derivative of <s_i | s_j>
                        D_om[x][4*iat][4*jat] = \
                        - ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) ) * \
                          (2.0*r) * d[x] * sji * amp[iat] * amp[jat]                                \
                        - 2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) * sji * amp[iat] *   \
                                                                                       amp[jat] *   \
                          ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) )

                        # Derivative of <s_i | p_j>
                        stv = np.sqrt(8.0) * rcov[jat] * r * sji
                        for i_sp in range(3):
                            D_om[x][4*iat][4*jat+i_sp+1] = \
                            ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) ) \
                            * stv * amp[iat] * amp[jat] * ( FingerPrint.kron_delta(x, i_sp)         \
                                                           - np.dot( d[x], d[i_sp] ) * 2.0 * r )    \
                          - 2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) * stv * d[i_sp] *  \
                                                                              amp[iat] * amp[jat] * \
                            ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) )

                        # Derivative of <p_i | s_j>
                        stv = np.sqrt(8.0) * rcov[iat] * r * sji * -1.0
                        for i_ps in range(3):
                            D_om[x][4*iat+i_ps+1][4*jat] = \
                            ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) ) \
                            * stv * amp[iat] * amp[jat] * ( FingerPrint.kron_delta(x, i_ps)         \
                                                           - np.dot( d[x], d[i_ps] ) * 2.0 * r )    \
                          - 2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) * stv * d[i_ps] *  \
                                                                              amp[iat] * amp[jat] * \
                            ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) )

                        # Derivative of <p_i | p_j>
                        stv = - 8.0 * rcov[iat] * rcov[jat] * r * r * sji
                        for i_pp in range(3):
                            for j_pp in range(3):
                                D_om[x][4*iat+i_pp+1][4*jat+j_pp+1] = \
                            ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) ) \
                            * d[x] * stv * amp[iat] * amp[jat] *                                    \
                            ( FingerPrint.kron_delta(x, j_pp) - 2.0 * r * d[i_pp] * d[j_pp] ) +     \
                            ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) ) \
                            * stv * amp[iat] * amp[jat] * ( FingerPrint.kron_delta(x, i_pp) *       \
                                              d[j_pp] + FingerPrint.kron_delta(x, j_pp) * d[i_pp] ) \
                          - 2.0 * NC * fc * dnc[x] * (1.0 - dnc2 * fc)**(NC - 1) * stv *            \
                            ( np.dot(d[i_pp], d[j_pp]) -                                            \
                                                       FingerPrint.kron_delta(i_pp, j_pp) * 0.5/r ) \
                            * amp[iat] * amp[jat] *                                                 \
                            ( FingerPrint.kron_delta(iat, D_n) - FingerPrint.kron_delta(jat, D_n) )

        return D_om
    
    def get_D_fp(self, lat, rxyz, types, znucl, x, D_n, iat):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        curoff = self.cutoff
        if lmax == 0:
            lseg = 1
            # l = 1
        else:
            lseg = 4
            # l = 2
        amp, n_sphere, icenter, rxyz_sphere, rcov_sphere = \
                       self.get_sphere(lat, rxyz, types, znucl, iat)
        om = self.get_gom(rxyz_sphere, rcov_sphere, amp)
        lamda_om, Varr_om = np.linalg.eig(om)
        lamda_om = np.real(lamda_om)
        lamda_om_list = lamda_om.tolist()
        null_Varr = np.vstack( (np.zeros_like(Varr_om[:, 0]), ) ).T
        for n in range(nx*lseg - len(lamda_om_list)):
            lamda_om_list.append(0.0)
            Varr_om_new = np.hstack((Varr_om, null_Varr))
            Varr_om = Varr_om_new.copy()

        lamda_om = np.array(lamda_om_list, float)

        # Sort eigen_val & eigen_vec joint matrix in corresponding descending order of eigen_val
        lamda_Varr_om = np.vstack((lamda_om, Varr_om))
        sorted_lamda_Varr_om = lamda_Varr_om[ :, lamda_Varr_om[0].argsort()]
        sorted_Varr_om = sorted_lamda_Varr_om[1:, :]

        N_vec = len(sorted_Varr_om[0])
        D_fp = np.zeros((nx*lseg, 1)) + 1j*np.zeros((nx*lseg, 1))
        # D_fp = np.zeros((nx*lseg, 1))
        D_om = self.get_D_gom(rxyz_sphere, rcov_sphere, amp, D_n, icenter)
        if x == 0:
            Dx_om = D_om[0, :, :]
            for i in range(N_vec):
                Dx_mul_V_om = np.matmul(Dx_om, sorted_Varr_om[:, i])
                D_fp[i][0] = np.matmul(sorted_Varr_om[:, i].T, Dx_mul_V_om)
        elif x == 1:
            Dy_om = D_om[1, :, :]
            for j in range(N_vec):
                Dy_mul_V_om = np.matmul(Dy_om, sorted_Varr_om[:, j])
                D_fp[j][0] = np.matmul(sorted_Varr_om[:, j].T, Dy_mul_V_om)
        elif x == 2:
            Dz_om = D_om[2, :, :]
            for k in range(N_vec):
                Dz_mul_V_om = np.matmul(Dz_om, sorted_Varr_om[:, k])
                D_fp[k][0] = np.matmul(sorted_Varr_om[:, k].T, Dz_mul_V_om)
        else:
            print("Error: Wrong x value! x can only be 0,1,2")

        # D_fp = np.real(D_fp)
        # print("D_fp {0:d} = {1:s}".format(x, np.array_str(D_fp, precision=6, suppress_small=False)) )
        # D_fp_factor = np.zeros(N_vec)
        # D_fp_factor = np.zeros(N_vec) + 1j*np.zeros(N_vec)
        # for N in range(N_vec):
        #     D_fp_factor[N] = 1/D_fp[N][0]
        #     D_fp[N][0] = (np.exp( np.log(D_fp_factor[N]*D_fp[N][0] + 1.2) ) - 1.2)/D_fp_factor[N]
        return D_fp
    
    def get_D_fp_mat(self, lat, rxyz, types, znucl, iat):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        curoff = self.cutoff
        if lmax == 0:
            lseg = 1
            # l = 1
        else:
            lseg = 4
            # l = 2
        amp, n_sphere, icenter, rxyz_sphere, rcov_sphere = \
                      get_sphere(ntyp, nx, lmax, lat, rxyz, types, znucl, cutoff, iat)
        # om = get_gom(lseg, rxyz_sphere, rcov_sphere, amp)
        # lamda_om, Varr_om = np.linalg.eig(om)
        # lamda_om = np.real(lamda_om)
        # N_vec = len(Varr_om[0])
        nat = len(rxyz_sphere)
        D_fp_mat = np.zeros((3, nx*lseg, nat)) + 1j*np.zeros((3, nx*lseg, nat))
        for i in range(3*nat):
            D_n = i // 3
            x = i % 3
            D_fp = self.get_D_fp(lat, rxyz, types, znucl, x, D_n, iat)
            for j in range(len(D_fp)):
                D_fp_mat[x][j][D_n] = D_fp[j][0]
                # D_fp_mat[x, :, D_n] = D_fp
                # Another way to compute D_fp_mat is through looping np.column_stack((a,b))
        # print("D_fp_mat = \n{0:s}".format(np.array_str(D_fp_mat, precision=6, suppress_small=False)) )
        return D_fp_mat
    
    @staticmethod
    def get_fp_nonperiodic(rxyz, znucls):
        rcov = []
        amp = [1.0] * len(rxyz)
        for x in znucls:
            rcov.append(rcovdata.rcovdata[x][2])
        gom = get_gom(1, rxyz, rcov, amp)
        fp = np.linalg.eigvals(gom)
        fp = sorted(fp)
        fp = np.array(fp, float)
        return fp
    
    @staticmethod
    def get_fpdist_nonperiodic(fp1, fp2):
        d = fp1 - fp2
        return np.sqrt(np.vdot(d, d))
    
    @staticmethod
    def get_fpdist(self, types, fp1, fp2, mx = False):
        ntyp = self.ntyp
        nat, lenfp = np.shape(fp1)
        fpd = 0.0
        for ityp in range(ntyp):
            itype = ityp + 1
            MX = np.zeros((nat, nat))
            for iat in range(nat):
                if types[iat] == itype:
                    for jat in range(nat):
                        if types[jat] == itype:
                            tfpd = fp1[iat] - fp2[jat]
                            MX[iat][jat] = np.sqrt(np.vdot(tfpd, tfpd))

            row_ind, col_ind = linear_sum_assignment(MX)
            # print(row_ind, col_ind)
            total = MX[row_ind, col_ind].sum()
            fpd += total

        fpd = fpd / nat
        if mx:
            return fpd, col_ind
        else:
            return fpd
        
    def get_fp(self, lat, rxyz, types, znucl, iat):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        curoff = self.cutoff
        if lmax == 0:
            lseg = 1
            l = 1
        else:
            lseg = 4
            l = 2
        # lfp = []
        sfp = []
        amp, n_sphere, icenter, rxyz_sphere, rcov_sphere = \
                       self.get_sphere(lat, rxyz, types, znucl, iat)
        # full overlap matrix
        nid = lseg * n_sphere
        gom = self.get_gom(rxyz_sphere, rcov_sphere, amp)
        val, vec = np.linalg.eig(gom)
        val = np.real(val)
        # fp0 = np.zeros(nx*lseg)
        fp0 = np.zeros((nx*lseg, 1))
        for i in range(len(val)):
            # fp0[i] = val[i]
            fp0[i][0] = val[i]
        fp0 = fp0/np.linalg.norm(fp0)
        # lfp = sorted(fp0)
        lfp = fp0[ fp0[ : , 0].argsort(), : ]
        # lfp.append(sorted(fp0))
        pvec = np.real(np.transpose(vec)[0])
        # contracted overlap matrix
        if contract:
            nids = l * (ntyp + 1)
            omx = np.zeros((nids, nids))
            for i in range(nid):
                for j in range(nid):
                    # print ind[i], ind[j]
                    omx[ind[i]][ind[j]] = omx[ind[i]][ind[j]] + pvec[i] * gom[i][j] * pvec[j]
            # for i in range(nids):
            #     for j in range(nids):
            #         if abs(omx[i][j] - omx[j][i]) > 1e-6:
            #             print ("ERROR", i, j, omx[i][j], omx[j][i])
            # print omx
            sfp0 = np.linalg.eigvals(omx)
            sfp.append(sorted(sfp0))

        # print ("n_sphere_min", min(n_sphere_list))
        # print ("n_shpere_max", max(n_sphere_list)) 

        if contract:
            # sfp = np.array(sfp, float)
            sfp = np.vstack( (np.array(sfp, float), ) ).T
            return sfp
        else:
            lfp = np.array(lfp, float)
            return lfp
        
    def get_fp_energy(self, lat, rxyz, types, znucl):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        curoff = self.cutoff
        fp_dist = 0.0
        fpdist_error = 0.0
        temp_num = 0.0
        temp_sum = 0.0
        for ityp in range(ntyp):
            itype = ityp + 1
            for i_atom in range(len(rxyz)):
                if types[i_atom] == itype:
                    for j_atom in range(len(rxyz)):
                        if types[j_atom] == itype:
                            fp_iat = self.get_fp(lat, rxyz, types, znucl, i_atom)
                            fp_jat = self.get_fp(lat, rxyz, types, znucl, j_atom)
                            dfp_ij = fp_iat - fp_jat
                            temp_num = fpdist_error + np.matmul(dfp_ij.T, dfp_ij)
                            temp_sum = fp_dist + temp_num
                            accum_error = temp_num - (temp_sum - fp_dist)
                            fp_dist = temp_sum


        # print ( "Finger print energy = {0:s}".format(np.array_str(fp_dist, \
        #                                            precision=6, suppress_small=False)) )
        return fp_dist
    
    def get_fp_forces(self, lat, rxyz, types, znucl):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        curoff = self.cutoff
        rxyz_new = rxyz.copy()
        # fp_dist = 0.0
        # fpdist_error = 0.0
        # fpdist_temp_sum = 0.0
        # fpdsit_temp_num = 0.0
        del_fp = np.zeros((len(rxyz_new), 3))
        sum_del_fp = np.zeros(3)
        # fp_dist = 0.0
        for k_atom in range(len(rxyz_new)):

            for i_atom in range(len(rxyz_new)):
                # del_fp = np.zeros(3)
                # temp_del_fp = np.zeros(3)
                # accum_error = np.zeros(3)
                # temp_sum = np.zeros(3)
                for j_atom in range(len(rxyz_new)):
                    fp_iat = self.get_fp(lat, rxyz_new, types, znucl, i_atom)
                    fp_jat = self.get_fp(lat, rxyz_new, types, znucl, j_atom)
                    D_fp_mat_iat = self.get_D_fp_mat(lat, rxyz_new, types, znucl, i_atom)
                    D_fp_mat_jat = self.get_D_fp_mat(lat, rxyz_new, types, znucl, j_atom)
                    diff_fp = fp_iat-fp_jat
                    kat_in_i_sphere, kat_i = \
                    self.get_common_sphere(lat, rxyz_new, types, znucl, k_atom, i_atom)
                    kat_in_j_sphere, kat_j = \
                    self.get_common_sphere(lat, rxyz_new, types, znucl, k_atom, j_atom)
                    if kat_in_i_sphere == True and kat_in_j_sphere == True:
                        diff_D_fp_x = D_fp_mat_iat[0, :, kat_i] - D_fp_mat_jat[0, :, kat_j]
                        diff_D_fp_y = D_fp_mat_iat[1, :, kat_i] - D_fp_mat_jat[1, :, kat_j]
                        diff_D_fp_z = D_fp_mat_iat[2, :, kat_i] - D_fp_mat_jat[2, :, kat_j]
                    elif kat_in_i_sphere == True and kat_in_j_sphere == False:
                        diff_D_fp_x = D_fp_mat_iat[0, :, kat_i]
                        diff_D_fp_y = D_fp_mat_iat[1, :, kat_i]
                        diff_D_fp_z = D_fp_mat_iat[2, :, kat_i]
                    elif kat_in_i_sphere == False and kat_in_j_sphere == True:
                        diff_D_fp_x = - D_fp_mat_jat[0, :, kat_j]
                        diff_D_fp_y = - D_fp_mat_jat[1, :, kat_j]
                        diff_D_fp_z = - D_fp_mat_jat[2, :, kat_j]
                    else:
                        diff_D_fp_x = np.zeros_like(D_fp_mat_jat[0, :, kat_j])
                        diff_D_fp_y = np.zeros_like(D_fp_mat_jat[1, :, kat_j])
                        diff_D_fp_z = np.zeros_like(D_fp_mat_jat[2, :, kat_j])

                    # Kahan sum implementation
                    '''
                    diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x)[::-1], ) ).T
                    diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y)[::-1], ) ).T
                    diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z)[::-1], ) ).T
                    temp_del_fp[0] = accum_error[0] + np.real( np.matmul( diff_fp.T,  diff_D_fp_x ) )
                    temp_del_fp[1] = accum_error[1] + np.real( np.matmul( diff_fp.T,  diff_D_fp_y ) )
                    temp_del_fp[2] = accum_error[2] + np.real( np.matmul( diff_fp.T,  diff_D_fp_z ) )
                    temp_sum[0] = del_fp[0] + temp_del_fp[0]
                    temp_sum[1] = del_fp[1] + temp_del_fp[1]
                    temp_sum[2] = del_fp[2] + temp_del_fp[2]
                    accum_error[0] = temp_del_fp[0] - (temp_sum[0] - del_fp[0])
                    accum_error[1] = temp_del_fp[1] - (temp_sum[1] - del_fp[1])
                    accum_error[2] = temp_del_fp[2] - (temp_sum[2] - del_fp[2])
                    del_fp[0] = temp_sum[0]
                    del_fp[1] = temp_sum[1]
                    del_fp[2] = temp_sum[2]
                    fp_iat = self.get_fp(lat, rxyz, types, znucl, i_atom)
                    fp_jat = self.get_fp(lat, rxyz, types, znucl, j_atom)
                    dfp_ij = fp_iat - fp_jat
                    fpdist_temp_num = fpdist_error + np.matmul(dfp_ij.T, dfp_ij)
                    fpdist_temp_sum = fp_dist + fpdist_temp_num
                    fpdist_error = fpdist_temp_num - (fpdist_temp_sum - fp_dist)
                    fp_dist = fpdist_temp_sum
                    '''


                    diff_D_fp_x = np.vstack( (np.array(diff_D_fp_x), ) ).T
                    diff_D_fp_y = np.vstack( (np.array(diff_D_fp_y), ) ).T
                    diff_D_fp_z = np.vstack( (np.array(diff_D_fp_z), ) ).T
                    # print("fp_dim", fp_iat.shape)
                    # print("diff_D_fp_x_dim", diff_D_fp_x.shape)

                    del_fp[i_atom][0] = del_fp[i_atom][0] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_x ) )
                    del_fp[i_atom][1] = del_fp[i_atom][1] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_y ) )
                    del_fp[i_atom][2] = del_fp[i_atom][2] + \
                                        2.0*np.real( np.matmul( diff_fp.T, diff_D_fp_z ) )
                    
                    # fp_iat = self.get_fp(lat, rxyz, types, znucl, i_atom)
                    # fp_jat = self.get_fp(lat, rxyz, types, znucl, j_atom)
                    # dfp_ij = fp_iat - fp_jat
                    # fp_dist = fp_dist + np.matmul(dfp_ij.T, dfp_ij)

                    # print("del_fp = ", del_fp)
                    # rxyz[i_atom] = rxyz[i_atom] - step_size*del_fp
                    '''
                    if max(del_fp) < atol:
                        print ("i_iter = {0:d} \nrxyz_final = \n{1:s}".\
                              format(i_iter+1, np.array_str(rxyz, precision=6, \
                              suppress_small=False)))
                        return
                        # with np.printoptions(precision=3, suppress=True):
                        # sys.exit("Reached user setting tolerance, program ended")
                    else:
                        print ("i_iter = {0:d} \nrxyz = \n{1:s}".\
                              format(i_iter, np.array_str(rxyz, precision=6, suppress_small=False)))
                    '''

                # rxyz_new[i_atom] = rxyz_new[i_atom] - step_size*del_fp/np.linalg.norm(del_fp)

            sum_del_fp = np.sum(del_fp, axis=0)
            for ii_atom in range(len(rxyz_new)):
                del_fp[ii_atom, :] = del_fp[ii_atom, :] - sum_del_fp/len(rxyz_new)
            '''
            print ( "i_iter = {0:d} \nrxyz_final = \n{1:s}".\
                  format(i_iter+1, np.array_str(rxyz_new, precision=6, suppress_small=False)) )
            print ( "Forces = \n{0:s}".\
                  format(np.array_str(del_fp, precision=6, suppress_small=False)) )
            print ( "Finger print energy difference = {0:s}".\
                  format(np.array_str(fp_dist, precision=6, suppress_small=False)) )
            '''

        return del_fp
    
    def get_FD_stress(self, lat, pos, types, znucl):
        contract = self.contract
        ntyp = self.ntyp
        nx = self.nx
        lmax = self.lmax
        curoff = self.cutoff
        rxyz = np.dot(pos, lat)
        fp_energy = 0.0
        fp_energy_new = 0.0
        fp_energy_left = 0.0
        fp_energy_right = 0.0
        # cell_vol = 0.0
        # lat_new = lat.copy()
        # lat_left = lat.copy()
        # lat_right = lat.copy()
        rxyz = np.dot(pos, lat)
        # rxyz_new = np.dot(pos, lat_new)
        # rxyz_left = np.dot(pos, lat_left)
        # rxyz_right = np.dot(pos, lat_right)
        rxyz_delta = np.zeros_like(rxyz)
        cell_vol = np.inner( lat[0], np.cross( lat[1], lat[2] ) )
        stress = np.zeros((3, 3))
        fp_energy = self.get_fp_energy(lat, rxyz, types, znucl)
        strain_delta = step_size*np.random.randint(1, 9999, (3, 3))/9999
        rxyz_ratio = np.diag(np.ones(3))
        rxyz_ratio_new = rxyz_ratio.copy()
        for m in range(3):
            for n in range(3):
                h = strain_delta[m][n]
                rxyz_ratio_left = np.diag(np.ones(3))
                rxyz_ratio_right = np.diag(np.ones(3))
                rxyz_ratio_left[m][n] = rxyz_ratio[m][n] - h
                rxyz_ratio_right[m][n] = rxyz_ratio[m][n] + h
                lat_left = np.multiply(lat, rxyz_ratio_left)
                lat_right = np.multiply(lat, rxyz_ratio_right)
                rxyz_left = np.dot(pos, lat_left)
                rxyz_right = np.dot(pos, lat_right)
                fp_energy_left = self.get_fp_energy(lat_left, rxyz_left, types, znucl)
                fp_energy_right = self.get_fp_energy(lat_right, rxyz_right, types, znucl)
                stress[m][n] = - (fp_energy_right - fp_energy_left)/(2.0*h*cell_vol)
        #################

        #################
        return stress