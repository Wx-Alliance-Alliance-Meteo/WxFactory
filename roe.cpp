
/*******************************************************************************
  Description: Innerhalb dieser Routine werden die primitiven Fluesse
               fuer den Low Mach low Dissipation Roe-Riemann-Löser (L2Roe
               berechnet. Die Funktion verwendet die links- und rechtsseitigen
               Variablen (st_left, st_right) an der betrachteten
               Kontrollvolumensgrenze, die in konservativer Form gegeben
               sein muessen. In f wird der Fluss zurueckgegeben, n beinhaltet
               die Normale.
*******************************************************************************/
const int MAX_CONSERVATIVE = 4; // Nombre dequations
void convective_fluxes_l2Roe(State_impl *st_left, State_impl *st_right,
                              double f[], double n[],
                              int shock_fix){
   double vlx, vly, vnl, vtl, vrx, vry, vnr, vtr;
   double cl, cr, MaL, MaR, fac;
   double Hl, Hr, pl, pr, inv_u_0;
   double rhoroe, Hroe, croe, sqrtrhol, sqrtrhor;
   double vroex, vroey, vnroe, v2roe;
   double deltalambda1, deltalambda2;
   double ul[MAX_CONSERVATIVE], ur[MAX_CONSERVATIVE];
   double lambda[MAX_CONSERVATIVE], diff[MAX_CONSERVATIVE];
   double alpha[MAX_CONSERVATIVE], diffusion[MAX_CONSERVATIVE];
   double roevec[MAX_CONSERVATIVE][MAX_CONSERVATIVE];
   int    i, j;

   for(i = 0; i < MAX_CONSERVATIVE; i++){
     ul[i] = st_left->data.cons[i];
     ur[i] = st_right->data.cons[i];
   }

   inv_u_0  = 1.0 / ul[IRHO];
   vlx = ul[IRHOVX] * inv_u_0; // Velocity (along x)
   vly = ul[IRHOVY] * inv_u_0; // Velocity (along y)
   vnl = vlx * n[0] + vly * n[1]; // Normal
   vtl = vly * n[0] - vlx * n[1]; // Tangent
   // Gamma = heat capacity ratio
   pl   = (GAMMA -1) * (ul[IRHOE] - 0.5 * inv_u_0 * (SQR(ul[IRHOVX])+SQR(ul[IRHOVY]))); // Pressure? Yes
   Hl   = (ul[IRHOE] + pl) * inv_u_0; // Enthalpy?
   cl = sqrt(GAMMA * pl * inv_u_0); // Speed of sound
   MaL = sqrt(SQR(vlx)+SQR(vly))/cl;

   inv_u_0  = 1.0 / ur[IRHO];
   vrx = ur[IRHOVX] * inv_u_0;
   vry = ur[IRHOVY] * inv_u_0;
   vnr = vrx * n[0] + vry * n[1];
   vtr = vry * n[0] - vrx * n[1];
   pr   = (GAMMA -1) * (ur[IRHOE] - 0.5 * inv_u_0 * (SQR(ur[IRHOVX])+SQR(ur[IRHOVY])));
   Hr   = (ur[IRHOE] + pr) * inv_u_0;
   cr = sqrt(GAMMA * pr * inv_u_0);
   MaR = sqrt(SQR(vrx)+SQR(vry))/cr;

   //Roe averages
   rhoroe = sqrt(ul[IRHO]*ur[IRHO]);
   sqrtrhol = sqrt(ul[IRHO]);
   sqrtrhor = sqrt(ur[IRHO]);
   vroex = (sqrtrhol*vlx + sqrtrhor*vrx)/(sqrtrhol + sqrtrhor);
   vroey = (sqrtrhol*vly + sqrtrhor*vry)/(sqrtrhol + sqrtrhor);
   vnroe = vroex * n[0] + vroey * n[1];
   v2roe = SQR(vroex) + SQR(vroey);
   Hroe = (sqrtrhol*Hl + sqrtrhor*Hr)/(sqrtrhol+sqrtrhor);
   croe = sqrt((GAMMA -1)*(Hroe-0.5*v2roe));

   lambda[IRHO] = fabs(vnroe-croe);
   lambda[IRHOVX] = fabs(vnroe);
   lambda[IRHOVY] = fabs(vnroe);
   lambda[IRHOE] = fabs(vnroe + croe);

   diff[IRHO] = ur[IRHO] - ul[IRHO];
   diff[IVX] = vnr - vnl;
   diff[IVY] = vtr - vtl;
   diff[IP] = pr - pl;

   //Entropy fix
   deltalambda1 = max((vnr-cr)-(vnl-cl),0.);
   deltalambda2 = max((vnr+cr)-(vnl+cl),0.);
   if(lambda[IRHO]<2*deltalambda1)
     lambda[IRHO] = SQR(lambda[IRHO])/(4*deltalambda1) + deltalambda1;
   if(lambda[IRHOE]<2*deltalambda2)
     lambda[IRHOE] = SQR(lambda[IRHOE])/(4*deltalambda2) + deltalambda2;

   //Shock fix
   if (shock_fix != 0)
     lambda[IRHOVX] = max(croe, sqrt(v2roe));

   //Low Mach fix
   fac = min(1., max(MaL, MaR));
   if (shock_fix == 0){
     diff[IVX] *= fac;
     diff[IVY] *= fac;
   }

   alpha[IRHO] = (diff[IP]/croe - rhoroe*diff[IVX])/(2*croe);
   alpha[IRHOVX] = diff[IRHO] - diff[IP]/(croe*croe);
   alpha[IRHOVY] = rhoroe*diff[IVY];
   alpha[IRHOE] = (diff[IP]/croe + rhoroe*diff[IVX])/(2*croe);

   //Roe vectors
   roevec[0][0]  = 1.;
   roevec[0][1]  = vroex - croe*n[0];
   roevec[0][2]  = vroey - croe*n[1];
   roevec[0][3]  = Hroe - croe*vnroe;

   roevec[1][0] = 1.;
   roevec[1][1] = vroex;
   roevec[1][2] = vroey;
   roevec[1][3] = 0.5*v2roe;

   roevec[2][0] = 0.;
   roevec[2][1] = -n[1];
   roevec[2][2] = n[0];
   roevec[2][3] = vroey * n[0] - vroex * n[1];

   roevec[3][0] = 1.;
   roevec[3][1] = vroex + croe*n[0];
   roevec[3][2] = vroey + croe*n[1];
   roevec[3][3] = Hroe + croe*vnroe;

   //left and right physical fluxes
   fac = ul[IRHO] * vnl;
   f[IRHO]   = fac;
   f[IRHOVX] = fac * vnl + pl;
   f[IRHOVY] = fac * vtl;
   f[IRHOE]  = fac * Hl;

   fac = ur[IRHO] * vnr;
   f[IRHO]   += fac;
   f[IRHOVX] += fac * vnr + pr;
   f[IRHOVY] += fac * vtr;
   f[IRHOE]  += fac * Hr;

   //Numerical diffusion
   for (i=0; i<MAX_CONSERVATIVE; i++)
     for (j=0; j<MAX_CONSERVATIVE; j++)
       diffusion[i] = 0.;
   for (i=0; i<MAX_CONSERVATIVE; i++){
     fac = alpha[i]*lambda[i];
     for (j=0; j<MAX_CONSERVATIVE; j++)
       diffusion[j] += fac*roevec[i][j];
   }

   f[IRHO]   -= diffusion[IRHO];
   fac        = f[IRHOVX] * n[0] - f[IRHOVY] * n[1];
   f[IRHOVY]  = f[IRHOVX] * n[1] + f[IRHOVY] * n[0];
   f[IRHOVX]  = fac - diffusion[IRHOVX];
   f[IRHOVY] -= diffusion[IRHOVY];
   f[IRHOE]  -= diffusion[IRHOE];

   for(i = 0; i < MAX_CONSERVATIVE; i++)
     f[i] *= 0.5;
} /* end convective_fluxes_l2Roe */


