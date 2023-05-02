#define _USE_MATH_DEFINES
#include <iostream>
#include <omp.h>
#include <time.h>  
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;

//#define N 902     // počet uzlov
//#define D 500
//#define R 6378    // polomer Zeme 

#define N 8102
#define D 200000.0
#define R 6378000.0
     
// #define N 160002
// #define D 50000
// #define R 6378000

#define GM 398600.5   // geocentrická gravitačná konštanta
#define maxIter 100   // max. pocet iteracii
#define tol 0.0000001  // tolerancia 



int main(int argc, char** argv) {

    // inicializacia paralelneho programu // 
    
    int threads = 3;
    omp_set_num_threads(threads); // set number of threads 
    
    // alokacia vektorov
    double* B = new double[N] {0.0};  //zemepisná šírka
    double* L = new double[N] {0.0};  //zemepisná dĺžka
    double* H = new double[N] {0.0};  //výška
    double* g = new double[N] {0.0};  //g
    double* un = new double[N] {0.0}; //druhá derivácia u podľa normály
    double* XX = new double[N] {0.0};
    double* XY = new double[N] {0.0};
    double* XZ = new double[N] {0.0}; // súradnice bodov na Zemi XX, XY, XZ
    double* SX = new double[N] {0.0};
    double* SY = new double[N] {0.0};
    double* SZ = new double[N] {0.0};  // súradnice bodov na fiktívnej hranici SX, SY, SZ

    double* nx = new double[N] {0.0};
    double* ny = new double[N] {0.0};
    double* nz = new double[N] {0.0};  // súradnice norm. vektora


    std::ifstream myfile;
    std::string mystring;

    //myfile.open("BL-902.dat");
    myfile.open("BL-8102.dat");
    // myfile.open("BL-160002.dat");

    if (myfile.is_open())
    {
        for (int i = 0; i < N; i++)
        {
            myfile >> B[i];
            myfile >> L[i];
            myfile >> H[i];
            myfile >> g[i];   // right side
            myfile >> un[i];
        }
    }
    myfile.close();

    // SET COORDINATES  Xi, Si // 
    for (int i = 0; i < N; i++)
    {
        // Suradnice zeme
        //XX[i] = R * std::cos(B[i] * (M_PI / 180.0)) * std::cos(L[i] * (M_PI / 180.0));    
        XX[i] = (R + H[i]) * std::cos(B[i] * (M_PI / 180.0)) * std::cos(L[i] * (M_PI / 180.0));
        //XY[i] = R * std::cos(B[i] * (M_PI / 180.0)) * std::sin(L[i] * (M_PI / 180.0));
        XY[i] = (R + H[i]) * std::cos(B[i] * (M_PI / 180.0)) * std::sin(L[i] * (M_PI / 180.0));
        //XZ[i] = R * std::sin(B[i] * (M_PI / 180.0));
        XZ[i] = (R + H[i]) * std::sin(B[i] * (M_PI / 180.0));

        // Suradnice na fiktivnej hranici
        //SX[i] = (R - D) * std::cos(B[i] * (M_PI / 180.0)) * std::cos(L[i] * (M_PI / 180.0));    
        SX[i] = (R + H[i] - D) * std::cos(B[i] * (M_PI / 180.0)) * std::cos(L[i] * (M_PI / 180.0));
        //SY[i] = (R - D) * std::cos(B[i] * (M_PI / 180.0)) * std::sin(L[i] * (M_PI / 180.0));
        SY[i] = (R + H[i] - D) * std::cos(B[i] * (M_PI / 180.0)) * std::sin(L[i] * (M_PI / 180.0));
        //SZ[i] = (R - D) * std::sin(B[i] * (M_PI / 180.0));
        SZ[i] = (R + H[i] - D) * std::sin(B[i] * (M_PI / 180.0));

        // Norm vector
        nx[i] = -XX[i] / sqrt(XX[i] * XX[i] + XY[i] * XY[i] + XZ[i] * XZ[i]);
        ny[i] = -XY[i] / sqrt(XX[i] * XX[i] + XY[i] * XY[i] + XZ[i] * XZ[i]);
        nz[i] = -XZ[i] / sqrt(XX[i] * XX[i] + XY[i] * XY[i] + XZ[i] * XZ[i]);

    }

    // matica a
    double* aa = new double[N * N] {0.0};

    double ry = 0;
    double rx = 0;
    double rz = 0;
    double k = 0;
    double rNorm = 0;
    int i = 0; int j = 0; int ij = -1;

#pragma omp parallel for private(rx,ry,rz,rNorm,k,j,ij)  

    // COMPUTE Aij // 
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            rx = XX[i] - SX[j];
            ry = XY[i] - SY[j];
            rz = XZ[i] - SZ[j];

            k = rx * nx[i] + ry * ny[i] + rz * nz[i];
            rNorm = sqrt(rx * rx + ry * ry + rz * rz);

            ij = i * N + j;
            aa[ij] = k / (4 * M_PI * rNorm * rNorm * rNorm);

        }
        
    }

   
    // BCG ALGORITHM // 

    // zaciatocna podmienka
    double g0 = -GM / (R * R);

    // pomocne vektory
    double* s = new double[N] {0.0};
    double* p = new double[N] {0.0};
    double* t = new double[N] {0.0};
    double* v = new double[N] {0.0};

    double* x = new double[N] {0.0};
    double* r = new double[N] {0.0};   //vektor r
    double* r0 = new double[N] {0.0};  //rez


    // pomocne premenne 
    int iter = 0;
    double rhoPrev = 0.0, rhoAct = 0.0;
    double beta = 0.0, alfa = 0.0, omega = 0.0;
    double rv = 0.0;
    double normStemp = 0.0; double normS = 0.0;
    double ts = 0.0, tt = 0.0;
    double normR = 0.0;
    
    
    double resid = 0.0;


    // Začiatočné podmienky
    for (i = 0; i < N; i++) {
        x[i] = 0;
        //r[i] = g0;
        //r0[i] = g0;
        r[i] = -g[i] * 0.00001;
        r0[i] = -g[i] * 0.00001;
    }

    double norm = 0.0;
    for (int j = 0; j < N; j++) {
        norm += r0[j] * r0[j];
    }

    std::cout << iter << ": Norm of residuals: " << sqrt(norm) << "\n";
    norm = 0.0;
        

    // BCG algortihm // 
    for (iter = 1; iter < maxIter; iter++)
    {  
        // rhoPrev, rhoAct
        rhoPrev = rhoAct;
        rhoAct = 0.0;
        for (i = 0; i < N; i++)
        {
            rhoAct += r0[i] * r[i];
        }

        // Kontrola rhoAct
        if (rhoAct == 0.0) {
            printf("Method fails!!!\n");
            break;
        }

        // vektor p - prva iteracia
        if (iter == 1)                                                      // toto neviem ci tu zostava
        {
            for (int i = 0; i < N; i++)
            { 
               p[i] = r0[i]; 
            }
        }

        // vektor p - ostatne iteracie
        else 
        {   
            beta = (rhoAct / rhoPrev) * (alfa / omega);
            
            for (i = 0; i < N; i++)
            {
                p[i] = r0[i] + beta * (p[i] - omega * v[i]);
            }
        }

        // vektor v
//#pragma omp parallel for private(j,ij)
        for (i = 0; i < N; i++)
        {
            v[i] = 0.0;
            for (j = 0; j < N; j++)
            {
                ij = i * N + j;
                v[i] += aa[ij] * p[j];
            }
        }

        // alfa
        rv = 0.0;
        for (i = 0; i < N; i++)
        {
            rv += r[i] * v[i];
        }
        alfa = rhoAct / rv;

        // vektor s, norm of rez
        normStemp = 0.0;
        for (i = 0; i < N; i++)
        {
            s[i] = r0[i] - alfa * v[i];
            normStemp += s[i] * s[i];
        }
        normS = sqrt(normStemp);

        // kontrola norm of rez, vektor x
        if (normS < tol)
        {
            for (i = 0; i < N; i++)
            {
                x[i] += alfa * p[i];
            }
            break;
        }
//#pragma omp parallel for private(j,ij)   
        // vektor t
        for (i = 0; i < N; i++)
        {
          t[i] = 0.0;
          for (j = 0; j < N; j++)
            {
                ij = i * N + j;
                t[i] += aa[ij] * s[j];
            }
        }

        // omega
        ts = 0.0;
        tt = 0.0;
        for (i = 0; i < N; i++)
        {
            ts += t[i] * s[i];
            tt += t[i] * t[i];
        }
        omega = ts / tt;

        // vektor rez, norm of rez
        normR = 0.0;
        for (i = 0; i < N; i++)
        {
            x[i] += alfa * p[i] + omega * s[i];    // update solution x
            r0[i] = s[i] - omega * t[i];           // compute new residuum vector
            normR += r0[i] * r0[i];                // compute residuum norm
        }
        resid = sqrt(normR);

        printf("%d: Norm of residuals: %.20lf\n",iter, resid);
            
        // ukoncenie cyklu
        if (resid < tol || omega == 0) 
        {
            break;
        }

    }
   
    //deallokovanie vektorov
    delete[] r0; delete[] r; 
    delete[] v; delete[] p; 
    delete[] s; delete[] t;

    //vektor u - reálne dáta
    double* u = new double[N] {0.0};
    double tempu = 0;
    for (i = 0; i < N; i++) 
    {

        tempu = 0;
        for (int j = 0; j < N; j++) {
            rx = XX[i] - SX[j];
            ry = XY[i] - SY[j];
            rz = XZ[i] - SZ[j];

            rNorm = sqrt(rx * rx + ry * ry + rz * rz);
            tempu += (1 * x[j]) / (4 * M_PI * rNorm);
        }
        u[i] = tempu;
    }
    
    
    // FILE WRITE // 

    fstream myfileW;
    
    //myfileW.open("Output902.dat", std::ios::out);
    myfileW.open("Output8102.dat", std::ios::out);
    //myfileW.open("Output160002.dat", std::ios::out);
    
    if (!myfileW)
    {
        std::cout << "File not created\n";
    }
    else
    {
        for (int i = 0; i < N; i++)
        {
            myfileW << B[i] << " " << L[i] << " " << u[i] << "\n";
        }
        myfileW.close();
    
        std::cout << "File created succesful\n";
    
    }


    //deallokovanie vektorov
    delete[] B; delete[] L; delete[] H; delete[] g; delete[] un; delete[] u; delete[] x;
    delete[] SX; delete[] SY; delete[] SZ;
    delete[] XX; delete[] XY; delete[] XZ;
    delete[] nx; delete[] ny; delete[] nz;
    delete[] aa;
    
}