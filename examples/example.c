#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "CNetwork.h"
#include <gsl/gsl_sf_zeta.h>

long int N;
#define Number_Of_Creation 50
#define Number_Of_Bins 20
#define Number_Of_centrality 12

#define max_steps 100
#define MAX_WALKS 50

typedef struct {
    double alpha;     /* power-law exponent a */
    long int kmin;    /* lower cutoff */
    double ks;        /* KS distance for the chosen fit */
    int n_tail;       /* number of data points with k >= kmin */
    int status;       /* 0 = success, nonzero = failure */
} PowerLawFit;

/* Compare long ints for qsort */
static int cmp_long_int(const void *a, const void *b)
{
    long int x = *(const long int *)a;
    long int y = *(const long int *)b;

    if (x < y) return -1;
    if (x > y) return  1;
    return 0;
}

/* Log-likelihood for discrete power law:
   L(a) = -n*log(zeta(a,kmin)) - a*sum(log(k_i))
*/
static double discrete_loglik(double a, long int kmin, int n_tail, double sumlog)
{
    if (a <= 1.0) return -INFINITY;

    double z = gsl_sf_hzeta(a, (double)kmin);
    if (!isfinite(z) || z <= 0.0) return -INFINITY;

    return -((double)n_tail) * log(z) - a * sumlog;
}

/* Golden-section search to maximize the log-likelihood on [a_lo, a_hi] */
static double maximize_alpha(long int kmin, int n_tail, double sumlog,
                             double a_lo, double a_hi)
{
    const double gr = 0.6180339887498948482; /* (sqrt(5)-1)/2 */
    const int max_iter = 200;
    const double tol = 1e-8;

    double a = a_lo;
    double b = a_hi;

    double c = b - gr * (b - a);
    double d = a + gr * (b - a);

    double fc = discrete_loglik(c, kmin, n_tail, sumlog);
    double fd = discrete_loglik(d, kmin, n_tail, sumlog);

    for (int iter = 0; iter < max_iter; iter++) {
        if (fabs(b - a) < tol * (fabs(c) + fabs(d) + 1.0)) {
            break;
        }

        if (fc > fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - gr * (b - a);
            fc = discrete_loglik(c, kmin, n_tail, sumlog);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + gr * (b - a);
            fd = discrete_loglik(d, kmin, n_tail, sumlog);
        }
    }

    /* Return the best point among the bracket endpoints and interior points */
    double best_a = a;
    double best_ll = discrete_loglik(a, kmin, n_tail, sumlog);

    double ll_b = discrete_loglik(b, kmin, n_tail, sumlog);
    if (ll_b > best_ll) { best_ll = ll_b; best_a = b; }

    if (fc > best_ll) { best_ll = fc; best_a = c; }
    if (fd > best_ll) { best_ll = fd; best_a = d; }

    return best_a;
}

/* KS distance for a discrete power law, conditional on k >= kmin.
   Data array d[] must be sorted ascending, and start is the first index
   with d[start] == kmin.
*/
static double ks_distance_discrete(const long int *d, int start, int m,
                                   double alpha, long int kmin)
{
    double zmin = gsl_sf_hzeta(alpha, (double)kmin);
    if (!isfinite(zmin) || zmin <= 0.0) return INFINITY;

    int ntail = m - start;
    int cum = 0;
    double D = 0.0;

    for (int i = start; i < m; ) {
        long int k = d[i];
        int j = i;

        while (j + 1 < m && d[j + 1] == k) {
            j++;
        }

        cum += (j - i + 1);

        double emp_cdf = (double)cum / (double)ntail;

        /* Model CDF: P(K <= k | K >= kmin)
           = 1 - zeta(alpha, k+1) / zeta(alpha, kmin)
        */
        double model_cdf = 1.0 - gsl_sf_hzeta(alpha, (double)k + 1.0) / zmin;

        double diff = fabs(emp_cdf - model_cdf);
        if (diff > D) D = diff;

        i = j + 1;
    }

    return D;
}

/* Main fitting function:
   deg[] = out-degrees of all nodes
   N     = number of nodes
*/
PowerLawFit fit_outdegree_powerlaw_discrete(const long int *deg, int N)
{
    PowerLawFit out;
    out.alpha = NAN;
    out.kmin = -1;
    out.ks = INFINITY;
    out.n_tail = 0;
    out.status = 1;

    if (!deg || N <= 0) {
        return out;
    }

    /* Keep only positive degrees, because kmin must be >= 1 */
    long int *pos = (long int *)malloc((size_t)N * sizeof(long int));
    if (!pos) {
        return out;
    }

    int m = 0;
    for (int i = 0; i < N; i++) {
        if (deg[i] > 0) {
            pos[m++] = deg[i];
        }
    }

    if (m < 2) {
        free(pos);
        return out;
    }

    qsort(pos, (size_t)m, sizeof(long int), cmp_long_int);

    /* Prefix sum of logs for fast tail sumlog queries */
    double *pref_log = (double *)malloc((size_t)(m + 1) * sizeof(double));
    if (!pref_log) {
        free(pos);
        return out;
    }

    pref_log[0] = 0.0;
    for (int i = 0; i < m; i++) {
        pref_log[i + 1] = pref_log[i] + log((double)pos[i]);
    }

    double best_ks = INFINITY;
    double best_alpha = NAN;
    long int best_kmin = -1;
    int best_ntail = 0;

    /* Candidate kmin values are the observed positive degree values */
    for (int start = 0; start < m; ) {
        long int kmin = pos[start];

        /* Find end of this block of equal values */
        int end = start;
        while (end + 1 < m && pos[end + 1] == kmin) {
            end++;
        }

        int ntail = m - start;
        if (ntail >= 2) {
            double sumlog = pref_log[m] - pref_log[start];

            /* Search alpha on a reasonable interval */
            double alpha_hat = maximize_alpha(kmin, ntail, sumlog, 1.0001, 50.0);

            if (isfinite(alpha_hat) && alpha_hat > 1.0) {
                double ks = ks_distance_discrete(pos, start, m, alpha_hat, kmin);

                if (ks < best_ks) {
                    best_ks = ks;
                    best_alpha = alpha_hat;
                    best_kmin = kmin;
                    best_ntail = ntail;
                }
            }
        }

        start = end + 1;
    }

    free(pref_log);
    free(pos);

    if (isfinite(best_alpha) && best_kmin > 0) {
        out.alpha = best_alpha;
        out.kmin = best_kmin;
        out.ks = best_ks;
        out.n_tail = best_ntail;
        out.status = 0;
    }

    return out;
}

void work(double max_val,double min_val,long int *bins,long double **ad_mat,long double * property)
{
    /*
    for (int i = 0; i < N; i++) {
        fprintf(matrix1, "%Lf\n",property[i]);
    }
    */
    compute_histogram(property, N, bins,Number_Of_Bins, 0, 1);

    empty_vec(property);
    Pdf(property,bins,Number_Of_Bins,N);
    
}

void graph(long double *P, double min_val, double max_val, const char *network,const char *centrality,double shift) {
    //shift = 0;
    if (P == NULL) {
        fprintf(stderr, "Error: Data array P is NULL.\n");
        return;
    }

    FILE *gnuplot = _popen("\"C:\\Program Files\\gnuplot\\bin\\gnuplot.exe\" -persist", "w");
    if (gnuplot == NULL) {
        fprintf(stderr, "Error: Could not open Gnuplot. Check the executable path.\n");
        return;
    }

    fprintf(gnuplot, "set title '%s'\n", network);
    fprintf(gnuplot, "set xlabel '%s'\n",centrality);
    fprintf(gnuplot, "set ylabel 'Probability P(c)'\n");
    //fprintf(gnuplot, "set logscale xy\n"); // Set both axes to logarithmic scale
    fprintf(gnuplot, "set grid\n");
    
    

    // Reconstruct the bin width using the bounds provided
    double bin_width = (max_val - min_val) / (double)Number_Of_Bins;

    fprintf(gnuplot, "set boxwidth %lf\n", bin_width);
    fprintf(gnuplot, "set style fill solid 1.0 border -1\n");

    fprintf(gnuplot, "plot '-' with boxes lc rgb '#87CEEB' title 'Distribution'\n");


    // Send data to Gnuplot
    for (int i = 0; i < Number_Of_Bins; i++) {
        if (P[i] > 0.0) {
            // Calculate the exact center of the bin for the x-axis
            double bin_center = ((i + 0.5) * bin_width);
            fprintf(gnuplot, "%lf %Lf\n", (bin_center - shift), P[i]);
        }
    }

    fprintf(gnuplot, "e\n");
    fflush(gnuplot);
    _pclose(gnuplot);
}
int main()
{
    double max_val = -INFINITY;
    double min_val = INFINITY;
    long int src, dst;
    double w;
    long int max_node = -1;


    FILE *bar1 = fopen("bar1.txt", "w");
    FILE *matrix1 = fopen("matrix1.txt", "w");
    if (!bar1) return 1;

    char line[500];
    char line2[500];

    omp_set_num_threads(10);
    // =================================================================
    // 1. REAL NETWORK
    // =================================================================



    FILE *fp = fopen("data7.csv", "r");
    if (fp == NULL) {
        printf("Error: Cannot open file.\n");
        return 1;
    }

    // اسکن اول برای یافتن N
    while(fscanf(fp, "%ld,%ld,%lf", &src, &dst, &w) == 3) {
        if(src > max_node) max_node = src;
        if(dst > max_node) max_node = dst;
    }
    N = max_node;
    N++;//for zero indexing

    
    

    

    long double **Shannon_Entropy = (long double **) calloc(5, sizeof(long double *));
    for (long int i = 0; i < 5; i++) {
        Shannon_Entropy[i] = (long double *) calloc(Number_Of_centrality, sizeof(long double));
    }


    long int *bins = (long int *)calloc(Number_Of_Bins, sizeof(long int));
    
    


    srand(time(NULL));

    // تخصیص حافظه (با calloc که خود به خود صفر می‌کند)
    long double **ad_mat_initial = (long double **) calloc(N, sizeof(long double *));
    for(int i = 0; i < N; i++)
    {
        ad_mat_initial[i] = (long double *) calloc(N, sizeof(long double));
    }

    // اسکن دوم برای پر کردن ad_mat
    rewind(fp);
    while(fscanf(fp, "%ld,%ld,%lf", &src, &dst, &w) == 3) {
        ad_mat_initial[src][dst] = 1; //for zero index   //ad_mat_initial[src-1][dst-1] = 1;// for 1 index 
    }
    fclose(fp);

    
    long int isolated;
    for (long int i = 0; i < N; i++) {
        isolated = 0;
        for (long int j = 0; j < N; j++) {
            if (ad_mat_initial[i][j] == 0 && ad_mat_initial[j][i]  == 0 && i!= j) {
                isolated++;
            }
        }
        if(isolated == (N-1))
        {
            // Delete row r: shift rows upward
            for (long int j = i; j < (N - 1); j++) {
                for (long int k = 0; k < N; k++) {
                    ad_mat_initial[j][k] = ad_mat_initial[j + 1][k];
                }
            }
        
            // Delete column c: shift columns left
            for (long int j = 0; j < (N - 1); j++) {
                for (long int k = i; k < (N - 1); k++) {
                    ad_mat_initial[j][k] = ad_mat_initial[j][k + 1];
                }
            }
            N--;
        }
    }
    

    long int E = 0;
    // ۱. شمارش تعداد یال‌ها (E)
    for (long int i = 0; i < N; i++) {
        for (long int j = 0; j < N; j++) {
            if (ad_mat_initial[i][j] > 0.0) {
                E++;
            }
        }
    }
    ////////////////////////////////////////////////////
    // generating line dgraph

    long double **A_L = (long double **) calloc(E, sizeof(long double *));
    for(int i = 0; i < E; i++)
    {
        A_L[i] = (long double *) calloc(E, sizeof(long double));
    }
    generateLineDigraph(ad_mat_initial,E,A_L);
    for(long int i = 0;i < N;i++)
    {
        free(ad_mat_initial[i]);
    }
    free(ad_mat_initial);

    long double **ad_mat = (long double **) calloc(E, sizeof(long double *));
    for(int i = 0; i < E; i++)
    {
        ad_mat[i] = (long double *) calloc(E, sizeof(long double));
    }
    for(long int i = 0;i < E;i++)
    {
        for(long int j = 0;j < E;j++)
        {
            ad_mat[i][j] = A_L[i][j];
        }
    }
    for(long int i = 0;i < E;i++)
    {
        free(A_L[i]);
    }
    free(A_L);
    N = E;
    ///////////////////////////////////////////////

    E = 0;
    // ۱. شمارش تعداد یال‌ها (E)
    for (long int i = 0; i < N; i++) {
        for (long int j = 0; j < N; j++) {
            if (ad_mat[i][j] > 0.0) {
                E++;
            }
        }
    }

    printf("...........................................................\n");
    printf("Number of nodes (N): %ld\n", N);
    printf("Number of total edges (E): %ld\n",E);
    printf("...........................................................\n");


    long double *property = (long double *)calloc(N, sizeof(long double));
    long double *property1 = (long double *)calloc(N, sizeof(long double));
    long double *property2 = (long double *)calloc(N, sizeof(long double));
    long double *property3 = (long double *)calloc(N, sizeof(long double));
    long double *property4 = (long double *)calloc(N, sizeof(long double));
    long double *property5 = (long double *)calloc(N, sizeof(long double));
    long double *property6 = (long double *)calloc(N, sizeof(long double));
    long double *property7 = (long double *)calloc(N, sizeof(long double));
    long double *property8 = (long double *)calloc(N, sizeof(long double));
    long double *property9 = (long double *)calloc(N, sizeof(long double));
    long double *property10 = (long double *)calloc(N, sizeof(long double));
    long double *property11 = (long double *)calloc(N, sizeof(long double));
    long double *properties[] = {
        property,
        property1,
        property2,
        property3,
        property4,
        property5,
        property6,
        property7,
        property8,
        property9,
        property10,
        property11
    };

    long double **matrix = (long double **) calloc((5 * N), sizeof(long double *));
    for (long int i = 0; i < (5 * N); i++) {
        matrix[i] = (long double *) calloc(Number_Of_centrality, sizeof(long double));
    }
    double shift;

    const char *matrix_names[5] = {"Real", "Erdos", "BBCR", "Watts", "Config"};
    const char *centrality_name[Number_Of_centrality] = {"DC-in","DC-out","HCC-in","HCC-out","LC-in"
        ,"LC-out"
        ,"PC-in"
        ,"PC-out"
        ,"RWBC"
        ,"RWCC-in"
        ,"RWCC-out"
        ,"BC"
        };



    deg_in(ad_mat, property);
    deg_out(ad_mat, property1);
    harmonic_closenness_in(ad_mat, property2);
    harmonic_closenness_out(ad_mat, property3);
    leverage_in(ad_mat, property4);
    leverage_out(ad_mat, property5);
    PageRank_in(ad_mat, property6);
    PageRank_out(ad_mat, property7);
    Random_Walk_Betweenness_directed(ad_mat, property8,max_steps,MAX_WALKS);
    Harmonic_Random_Walk_Closeness_in(ad_mat, property9,max_steps,MAX_WALKS);
    Harmonic_Random_Walk_Closeness_out(ad_mat, property10,max_steps,MAX_WALKS);
    Betweenness_directed(ad_mat, property11);
    
    for(int i = 0;i < Number_Of_centrality;i++)
    {
        for(long int e = 0;e < Number_Of_Bins;e++)
        {
            bins[e] = 0;
        }
        min_val = INFINITY;
        max_val =  -INFINITY;
        for (int j = 0; j < N; j++) {
        if ((double)properties[i][j] < min_val) min_val = (double)properties[i][j];
        if ((double)properties[i][j] > max_val) max_val = (double)properties[i][j];
        }
        if(i == 4 || i == 5)
        {
            shift = -min_val / (max_val - min_val);
        }
        else
        {
            shift = 0;
        }
        normalize(properties[i],N);
        for(long int j = 0; j < N; j++)
        {
            matrix[j][i] = properties[i][j];
        }
        work(max_val,min_val,bins,ad_mat,properties[i]);
        Shannon_Entropy[0][i] = shannon_entropy(properties[i]);
        printf("%s Shannon Entropy for Real Network: %Lf\n",centrality_name[i], Shannon_Entropy[0][i]);
        graph(properties[i],0,1,"Real Network",centrality_name[i],shift);
    }


    

   // fprintf(out, "%f\n",shanon_entropy);


    ////////////////////////////////
    VisualizeGraph(ad_mat,N,"Real_dLine.png","lightblue","gray",0.5,0.8,1);




    // for out version
    long int *outdeg = (long int *)calloc(N, sizeof(long int));
    for(long int i = 0; i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            if(ad_mat[i][j] > 0)
            {
                outdeg[i]++;
            }
        }
    }
    PowerLawFit fit = fit_outdegree_powerlaw_discrete(outdeg, N);

    if (fit.status == 0) {
        printf("alpha = %.8f\n", fit.alpha);
        printf("kmin  = %ld\n", fit.kmin);
        printf("KS    = %.8f\n", fit.ks);
        printf("tail points = %d\n", fit.n_tail);
    } else {
        printf("Power-law fit failed.\n");
    }


    // ================================================================
    // 2. Real Netwrok Propertiese
    //=================================================================



   // ad_mat_real
    long double **ad_mat_real = (long double **) calloc(N, sizeof(long double *));
    for(int i = 0; i < N; i++)
    {
        ad_mat_real[i] = (long double *) calloc(N, sizeof(long double));
    }

    for(long int i = 0;i < N;i++)
    {
        for(long int j = 0; j < N;j++)
        {
            if(ad_mat[i][j] > 0)
            {
                ad_mat_real[i][j] = ad_mat[i][j];
            }
        }
    }

    // for out version
    double g = 0;
    double *k = (double *)calloc(N, sizeof(double));
    for(long int i = 0; i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            if(ad_mat[i][j] > 0)
            {
                k[i]++;
            }
        }
    }
    for(long int i = 0; i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            g += (k[i]*k[i]);
        }
    }
    g/=N;
    double g_prime = 0;
    for(long int i = 0; i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            g_prime += k[i];
        }
    }
    g_prime/=N;
    g_prime *= g_prime;
    g /= g_prime;
    printf("g out real: %lf\n",g);


    //for in version
    for(long int i = 0; i < N;i++)
    {
        k[i] = 0;
    }
    for(long int i = 0; i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            if(ad_mat[j][i] > 0)
            {
                k[i]++;
            }
        }
    }
    g = 0;
    for(long int i = 0; i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            g += (k[i]*k[i]);
        }
    }
    g/=N;
    g_prime = 0;
    for(long int i = 0; i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            g_prime += k[i];
        }
    }
    g_prime/=N;
    g_prime *= g_prime;
    g /= g_prime;
    printf("g in real: %lf\n",g);


    long double Average_Degree_in_double = 0;
    long int Average_Degree_in;
    for(long int i = 0; i < N; i++)
    {
        for(long int j = 0; j < N; j++)
        {
            if(ad_mat[j][i] > 0)
            {
                Average_Degree_in_double++;
            }
        }
    }
    Average_Degree_in_double /= N;
    long double Avg_deg_in = Average_Degree_in_double;
    
    


    long double Average_Degree_out_double = 0;
    long int Average_Degree_out;
    for(long int i = 0; i < N; i++)
    {
        for(long int j = 0; j < N; j++)
        {
            if(ad_mat[i][j] > 0)
            {
                Average_Degree_out_double++;
            }
        }
    }
    Average_Degree_out_double /= N;
    long double Avg_deg_out = Average_Degree_out_double;
    Average_Degree_out_double = round(Average_Degree_out_double/2);
    Average_Degree_out = (long int)Average_Degree_out_double;
    Average_Degree_out*=2;

    if((Average_Degree_out % 2) != 0)
    {
        Average_Degree_out--;
    }

    printf("Avg_deg_in: %Lf\n",Avg_deg_in);
    printf("Avg_deg_out: %Lf\n",Avg_deg_out);
    
    /*
    if (Average_Degree_out % 2 != 0)
    {
        Average_Degree_out--;
    }
    */
    
    

    long int zero_out_degree = 0;
    long int l;
    for(long int i = 0; i < N; i++)
    {
        l = 0;
        for(long int j = 0; j < N; j++)
        {
            if(ad_mat[i][j] > 0)
            {
                l++;
            }
        }
        if(l == 0)
        {
            zero_out_degree++;
        }
    }

    printf("zero_out_degree: %ld\n",zero_out_degree);
    long int zero_in_degree = 0;
    for(long int i = 0; i < N; i++)
    {
        l = 0;
        for(long int j = 0; j < N; j++)
        {
            if(ad_mat[j][i] > 0)
            {
                l++;
            }
        }
        if(l == 0)
        {
            zero_in_degree++;
        }
    }

    printf("zero_in_degree: %ld\n",zero_in_degree);

    // =================================================================
    // 3. ERDOS RENYI NETWORK (Equivalent)
    // =================================================================


    printf("===========================================\n");

    double shanon_entropy = 0;
    
    for(int i = 0;i < Number_Of_centrality;i++)
    {
        empty_vec(properties[i]);
    }
    for(long int e = 0;e < Number_Of_Bins;e++)
    {
        bins[e] = 0;
    }


    max_val = -INFINITY;
    min_val = INFINITY;
    long int current_edges = 0;
    for(int t = 0;t < Number_Of_Creation;t++)
    {
        empty_mat(ad_mat);
        Erdos_Renyi_directed_EQ(ad_mat,E);
    
    
        deg_in(ad_mat, property);
        deg_out(ad_mat, property1);
        harmonic_closenness_in(ad_mat, property2);
        harmonic_closenness_out(ad_mat, property3);
        leverage_in(ad_mat, property4);
        leverage_out(ad_mat, property5);
        PageRank_in(ad_mat, property6);
        PageRank_out(ad_mat, property7);
        Random_Walk_Betweenness_directed(ad_mat, property8,max_steps,MAX_WALKS);
        Harmonic_Random_Walk_Closeness_in(ad_mat, property9,max_steps,MAX_WALKS);
        Harmonic_Random_Walk_Closeness_out(ad_mat, property10,max_steps,MAX_WALKS);
        Betweenness_directed(ad_mat, property11);
        
        
        // ۱. شمارش تعداد یال‌ها 
        for (long int i = 0; i < N; i++) {
            for (long int j = 0; j < N; j++) {
                if (ad_mat[i][j] > 0.0) {
                    current_edges++;
                }
            }
        }
    }
    
    for(int i = 0;i < Number_Of_centrality;i++)
    {
        for(long int e = 0;e < Number_Of_Bins;e++)
        {
            bins[e] = 0;
        }
        min_val = INFINITY;
        max_val =  -INFINITY;
        for (int j = 0; j < N; j++) {
        if ((double)properties[i][j] < min_val) min_val = (double)properties[i][j];
        if ((double)properties[i][j] > max_val) max_val = (double)properties[i][j];
        }
        if(i == 4 || i == 5)
        {
            shift = -min_val / (max_val - min_val);
        }
        else
        {
            shift = 0;
        }
        normalize(properties[i],N);
        for(long int j = 0; j < N; j++)
        {
            matrix[(j+N)][i] = properties[i][j];
        }
        work(max_val,min_val,bins,ad_mat,properties[i]);
        Shannon_Entropy[1][i] = shannon_entropy(properties[i]);
        printf("%s Shannon Entropy for Erdos Network: %Lf\n",centrality_name[i], Shannon_Entropy[1][i]);
        graph(properties[i],0,1,"Erdos Network",centrality_name[i],shift);
    }
    
    VisualizeGraph(ad_mat,N,"Erdos_dLine.png","lightblue","gray",0.5,0.8,1);

    
    
    // =================================================================
    // 4. Bollobas-Borgs-Chayes-Riordan NETWORK (Equivalent)
    // =================================================================


    printf("===========================================\n");
    shanon_entropy = 0;
    for(int i = 0;i < Number_Of_centrality;i++)
    {
        empty_vec(properties[i]);
    }
    for(long int e = 0;e < Number_Of_Bins;e++)
    {
        bins[e] = 0;
    }
    

    long int weight_idx = 0;
    current_edges = 0;
    double alpha;//out degree node
    double beta;//add edge
    double gamma;//in degree node
    double delta_in;
    double delta_out = 0;

    

    beta = ((double)(E-N)/E);
    printf("beta: %f\n",beta);
    if(zero_out_degree == 0)
    {
        gamma = 0;
        alpha = (1- beta);
        printf("gamma: %f\n",gamma);
        printf("alpha: %f\n",alpha);
    }
    else
    {
        double A = (zero_in_degree/zero_out_degree);
        gamma = ((1-beta) / (1+A));
        printf("gamma: %f\n",gamma);
        alpha = (gamma*A);
        printf("alpha: %f\n",alpha);
    }

/*
    beta = ((double)(E-N)/E);
    printf("beta: %f\n",beta);
    double z = ((double)N/E);
    
    alpha = (1- ((double)1/(fit.alpha - 1)));
    printf("alpha: %f\n",alpha);
    

    gamma = z-alpha;
    printf("gamma: %f\n",gamma);
*/
    //delta_in = (((alpha + beta)*(fit.alpha-1)-1) / z);
    //printf("delta_in: %f\n",delta_in);
    delta_in = 50.0;
    delta_out = 50.0;

    Avg_deg_in = 0;
    Avg_deg_out = 0;
    max_val = -INFINITY;
    min_val = INFINITY;
    for(int ti = 0;ti < Number_Of_Creation;ti++)
    {
        empty_mat(ad_mat);
        Bollobas_Borgs_Chayes_Riordan_EQ(ad_mat,alpha,beta,gamma,delta_in,delta_out);


        



        deg_in(ad_mat, property);
        deg_out(ad_mat, property1);
        harmonic_closenness_in(ad_mat, property2);
        harmonic_closenness_out(ad_mat, property3);
        leverage_in(ad_mat, property4);
        leverage_out(ad_mat, property5);
        PageRank_in(ad_mat, property6);
        PageRank_out(ad_mat, property7);
        Random_Walk_Betweenness_directed(ad_mat, property8,max_steps,MAX_WALKS);
        Harmonic_Random_Walk_Closeness_in(ad_mat, property9,max_steps,MAX_WALKS);
        Harmonic_Random_Walk_Closeness_out(ad_mat, property10,max_steps,MAX_WALKS);
        Betweenness_directed(ad_mat, property11);
        
        //deg_in(ad_mat, property);
        /*
        for (long int i = 0; i < N; i++) {
            long int k = (long int)d_in[i];
            if (k < N) {
                property[k] += 1.0; 
            }
        }
        */
        
        for (long int i = 0; i < N; i++) {
            for (long int j = 0; j < N; j++) {
                if (ad_mat[i][j] > 0.0) {
                    current_edges++;
                }
            }
        }







        
        for(long int i = 0; i < N; i++)
        {
            for(long int j = 0; j < N; j++)
            {
                if(ad_mat[j][i] > 0)
                {
                    Avg_deg_in++;
                }
            }
        }
        
        
    
    
        for(long int i = 0; i < N; i++)
        {
            for(long int j = 0; j < N; j++)
            {
                if(ad_mat[i][j] > 0)
                {
                    Avg_deg_out++;
                }
            }
        }
    

    }
    Avg_deg_in /= (N*Number_Of_Creation);
    Avg_deg_out /= (N*Number_Of_Creation);
    printf("Avg_deg_in: %Lf\n",Avg_deg_in);
    printf("Avg_deg_out: %Lf\n",Avg_deg_out);



    for(int i = 0;i < Number_Of_centrality;i++)
    {
        for(long int e = 0;e < Number_Of_Bins;e++)
        {
            bins[e] = 0;
        }
        min_val = INFINITY;
        max_val =  -INFINITY;
        for (int j = 0; j < N; j++) {
        if ((double)properties[i][j] < min_val) min_val = (double)properties[i][j];
        if ((double)properties[i][j] > max_val) max_val = (double)properties[i][j];
        }
        if(i == 4 || i == 5)
        {
            shift = -min_val / (max_val - min_val);
        }
        else
        {
            shift = 0;
        }
        normalize(properties[i],N);
        for(long int j = 0; j < N; j++)
        {
            matrix[(j+(2*N))][i] = properties[i][j];
        }
        work(max_val,min_val,bins,ad_mat,properties[i]);
        Shannon_Entropy[2][i] = shannon_entropy(properties[i]);
        printf("%s Shannon Entropy for BBCR Network: %Lf\n",centrality_name[i], Shannon_Entropy[2][i]);
        graph(properties[i],0,1,"BBCR Network",centrality_name[i],shift);
    }



    // ۱. شمارش تعداد یال‌ها 
    current_edges/=Number_Of_Creation;
    printf("Number of edges: %ld\n",current_edges);

    VisualizeGraph(ad_mat,N,"BBCR_dLine.png","lightblue","gray",0.5,0.8,1);

    // =================================================================
    // 5. WATTS STROGATZ NETWORK (Equivalent)
    // =================================================================





    printf("===========================================\n");

    shanon_entropy = 0;
    for(int i = 0;i < Number_Of_centrality;i++)
    {
        empty_vec(properties[i]);
    }
    for(long int e = 0;e < Number_Of_Bins;e++)
    {
        bins[e] = 0;
    }


    

    double p_ws = 0.05; // احتمال بازسیم‌کشی (می‌توانید از کاربر دریافت کنید)

    Avg_deg_in = 0;
    Avg_deg_out = 0;
    int half = Average_Degree_out / 2;   // edges on each side

    //Average_Degree_out += 1;
    current_edges = 0;
    max_val = -INFINITY;
    min_val = INFINITY;
    double p0 = ((double)Average_Degree_out / (double)(N - 1));
    for(int t = 0;t < Number_Of_Creation;t++)
    {
        
        empty_mat(ad_mat);
        Watts_Strogatz_directed_EQ(ad_mat,p_ws,Average_Degree_out);

    
        deg_in(ad_mat, property);
        deg_out(ad_mat, property1);
        harmonic_closenness_in(ad_mat, property2);
        harmonic_closenness_out(ad_mat, property3);
        leverage_in(ad_mat, property4);
        leverage_out(ad_mat, property5);
        PageRank_in(ad_mat, property6);
        PageRank_out(ad_mat, property7);
        Random_Walk_Betweenness_directed(ad_mat, property8,max_steps,MAX_WALKS);
        Harmonic_Random_Walk_Closeness_in(ad_mat, property9,max_steps,MAX_WALKS);
        Harmonic_Random_Walk_Closeness_out(ad_mat, property10,max_steps,MAX_WALKS);
        Betweenness_directed(ad_mat, property11);
        
        
        for (long int i = 0; i < N; i++) {
            for (long int j = 0; j < N; j++) {
                if (ad_mat[i][j] > 0.0) {
                    current_edges++;
                }
            }
        }


        for(long int i = 0; i < N; i++)
        {
            for(long int j = 0; j < N; j++)
            {
                if(ad_mat[j][i] > 0)
                {
                    Avg_deg_in++;
                }
            }
        }
        
        
    
    
        for(long int i = 0; i < N; i++)
        {
            for(long int j = 0; j < N; j++)
            {
                if(ad_mat[i][j] > 0)
                {
                    Avg_deg_out++;
                }
            }
        }
    }
    Avg_deg_in /= (N*Number_Of_Creation);
    Avg_deg_out /= (N*Number_Of_Creation);
    printf("Avg_deg_in: %Lf\n",Avg_deg_in);
    printf("Avg_deg_out: %Lf\n",Avg_deg_out);
    
    for(int i = 0;i < Number_Of_centrality;i++)
    {
        for(long int e = 0;e < Number_Of_Bins;e++)
        {
            bins[e] = 0;
        }
        min_val = INFINITY;
        max_val =  -INFINITY;
        for (int j = 0; j < N; j++) {
        if ((double)properties[i][j] < min_val) min_val = (double)properties[i][j];
        if ((double)properties[i][j] > max_val) max_val = (double)properties[i][j];
        }
        if(i == 4 || i == 5)
        {
            shift = -min_val / (max_val - min_val);
        }
        else
        {
            shift = 0;
        }
        normalize(properties[i],N);
        for(long int j = 0; j < N; j++)
        {
            matrix[(j+(3*N))][i] = properties[i][j];
        }
        work(max_val,min_val,bins,ad_mat,properties[i]);
        Shannon_Entropy[3][i] = shannon_entropy(properties[i]);
        printf("%s Shannon Entropy for Watts Network: %Lf\n",centrality_name[i], Shannon_Entropy[3][i]);
        graph(properties[i],0,1,"Watts Network",centrality_name[i],shift);
    }



    current_edges /=Number_Of_Creation;
    
    printf("Number of edges: %ld\n",current_edges);

   // محاسبه هیستوگرام و رسم نمودار

    FILE *pipe1 = _popen("neato -Tpng -o graph1.png", "w");//FILE *pipe = _popen("neato -Tsvg -o graph.svg", "w");

    if (!pipe1) {
        perror("popen");
        return 1;
    }

    fprintf(pipe1, "digraph G {\n");
    fprintf(pipe1, "    graph [overlap=false, sep=\"+3\", dpi=300];\n");
    fprintf(pipe1, "    node [shape=circle, style=filled, fillcolor=lightblue, width=0.3, height=0.3, fontsize=8];\n");
    fprintf(pipe1, "    edge [color=gray, arrowsize=0.4];\n");

    // ONLY print edges. Do NOT print nodes individually.
    for (long int i = 0; i < N; i++) {
        for (long int j = 0; j < N; j++) {
            if (ad_mat[i][j] > 0) {
                fprintf(pipe1, "    %ld -> %ld;\n", i, j);
            }
        }
    }
    fprintf(pipe1, "}\n");


    VisualizeGraph(ad_mat,N,"Watts_dLine.png","lightblue","gray",0.5,0.8,1);



    // =================================================================
    // 6. CONFIGURATIONAL MODEL NETWORK (Equivalent)
    // =================================================================





    printf("===========================================\n");

    
    shanon_entropy = 0;
    for(int i = 0;i < Number_Of_centrality;i++)
    {
        empty_vec(properties[i]);
    }
    for(long int e = 0;e < Number_Of_Bins;e++)
    {
        bins[e] = 0;
    }

    long double **ad_mat_copy = (long double **) calloc(N, sizeof(long double *));
    for(long int i = 0; i < N; i++)
    {
        ad_mat_copy[i] = (long double *) calloc(N, sizeof(long double));
    }


    // Create multiple independent realizations
    max_val = -INFINITY;
    min_val = INFINITY;
    for (long int realization = 0; realization < Number_Of_Creation; realization++)
    {
        

        empty_mat(ad_mat);
        Configurational_model_directed_EQ(ad_mat_real,ad_mat_copy,E);



        // 5. Compute betweenness or other metrics
        deg_in(ad_mat_copy, property);
        deg_out(ad_mat_copy, property1);
        harmonic_closenness_in(ad_mat_copy, property2);
        harmonic_closenness_out(ad_mat_copy, property3);
        leverage_in(ad_mat_copy, property4);
        leverage_out(ad_mat_copy, property5);
        PageRank_in(ad_mat_copy, property6);
        PageRank_out(ad_mat_copy, property7);
        Random_Walk_Betweenness_directed(ad_mat_copy, property8,max_steps,MAX_WALKS);
        Harmonic_Random_Walk_Closeness_in(ad_mat_copy, property9,max_steps,MAX_WALKS);
        Harmonic_Random_Walk_Closeness_out(ad_mat_copy, property10,max_steps,MAX_WALKS);
        Betweenness_directed(ad_mat_copy, property11);
        
    }
    
    for(int i = 0;i < Number_Of_centrality;i++)
    {
        for(long int e = 0;e < Number_Of_Bins;e++)
        {
            bins[e] = 0;
        }
        min_val = INFINITY;
        max_val =  -INFINITY;
        for (int j = 0; j < N; j++) {
        if ((double)properties[i][j] < min_val) min_val = (double)properties[i][j];
        if ((double)properties[i][j] > max_val) max_val = (double)properties[i][j];
        }
        if(i == 4 || i == 5)
        {
            shift = -min_val / (max_val - min_val);
        }
        else
        {
            shift = 0;
        }
        normalize(properties[i],N);
        for(long int j = 0; j < N; j++)
        {
            matrix[(j+(4*N))][i] = properties[i][j];
        }
        work(max_val,min_val,bins,ad_mat,properties[i]);
        Shannon_Entropy[4][i] = shannon_entropy(properties[i]);
        printf("%s Shannon Entropy for Config Network: %Lf\n",centrality_name[i], Shannon_Entropy[4][i]);
        graph(properties[i],0,1,"Config Network",centrality_name[i],shift);
    }

    

    //writting properties
    for (int row = 0; row < (5*N); row++)
    {
        for (int col = 0; col < Number_Of_centrality; col++)
        {
            fprintf(matrix1, "%Lf", matrix[row][col]);
    
            if (col < Number_Of_centrality - 1)
                fprintf(matrix1, " ");
        }
    
        fprintf(matrix1, "\n");
    }

    //writing shannon entropies
    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < Number_Of_centrality; col++)
        {
            fprintf(bar1, "%Lf", Shannon_Entropy[row][col]);
    
            if (col < Number_Of_centrality - 1)
                fprintf(bar1, " ");
        }
    
        fprintf(bar1, "\n");
    }

    VisualizeGraph(ad_mat_copy,N,"Config_dLine.png","lightblue","gray",0.5,0.8,1);

    
    //free

    for(int i = 0; i < N; i++) {
        free(ad_mat[i]);
    }
    free(ad_mat);

    for(int i = 0; i < N; i++) {
        free(ad_mat_real[i]);
    }
    free(ad_mat_real);

    free(property);

    
    return 0;
}