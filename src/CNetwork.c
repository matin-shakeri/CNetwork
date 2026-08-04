#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>
extern long int N;


/////////////////////////////////////////////////////////////////////////////////
//                               Centrality                                    //
/////////////////////////////////////////////////////////////////////////////////


void deg_in(long double **ad_mat,long double *property)
{
    double *k_vec = (double *)calloc(N, sizeof(double)); 
    // محاسبه k_vec
    for(long int i = 0; i < N; i++)
    {
        for(long int j = 0; j < N; j++)
        {
            if(ad_mat[j][i] > 0)
            {
                k_vec[i]++;
            }
        }
    }

    
    for(long int e = 0; e < N;e++)
    {
        property[e] += k_vec[e];
    }
    


    
    free(k_vec);
    
}


void deg_out(long double **ad_mat,long double *property)
{
    double *k_vec = (double *)calloc(N, sizeof(double)); 
    // محاسبه k_vec
    for(long int i = 0; i < N; i++)
    {
        for(long int j = 0; j < N; j++)
        {
            if(ad_mat[i][j] > 0)
            {
                k_vec[i]++;
            }
        }
    }

    
    for(long int e = 0; e < N;e++)
    {
        property[e] += k_vec[e];
    }
    


    free(k_vec);
    
}


void Betweenness_directed(long double **ad_mat,long double *property)
{
    long double *BC = (long double *)calloc(N, sizeof(long double));
    long double *d = (long double *)malloc(N * sizeof(long double));     // فاصله‌ها
    long double *sigma = malloc(N * sizeof(long double)); // تعداد مسیرها
    long double *delta = (long double *)malloc(N * sizeof(long double)); // وابستگی
    long int *visited = (long int *)malloc(N * sizeof(long int));        // گره‌های دیده‌شده
    long int *S = (long int *)malloc(N * sizeof(long int));              // پشته (Stack)
    
    // ماتریس پیشروها (Predecessors)
    // P[w][v] = 1 یعنی v پیشرو w است
    long int **P = (long int **)malloc(N * sizeof(long int *));
    for (long int i = 0; i < N; i++) {
        P[i] = (long int *)malloc(N * sizeof(long int));
    }

   

    // =========================================================
    // حلقه اصلی: هر گره یک بار به عنوان مبدأ (start) انتخاب می‌شود
    // =========================================================
    for (long int start = 0; start < N; start++) {
        
        // 1. مقداردهی اولیه برای گره start فعلی
        long int stack_ptr = 0;
        for (long int i = 0; i < N; i++) {
            d[i] = INFINITY;
            sigma[i] = 0.0;
            visited[i] = 0;
            delta[i] = 0.0;
            for (long int j = 0; j < N; j++) {
                P[i][j] = 0; // پاک کردن پیشروها
            }
        }
        
        d[start] = 0.0;
        sigma[start] = 1.0; // یک مسیر به خودش وجود دارد

        // 2. مرحله Forward Pass (الگوریتم دایجسترای تغییر یافته)
        for (long int count = 0; count < N; count++) {
            // پیدا کردن گره با کمترین فاصله
            long double min_dist = INFINITY;
            long int current = -1;

            for (long int i = 0; i < N; i++) {
                if (!visited[i] && d[i] < min_dist) {
                    min_dist = d[i];
                    current = i;
                }
            }

            // اگر گرهی پیدا نشد یا بقیه غیرقابل دسترس هستند
            if (current == -1 || min_dist == INFINITY) break;

            visited[current] = 1;
            S[stack_ptr++] = current; // اضافه کردن گره به پشته

            // بررسی همسایه‌ها
            double EPSILON = (1e-12L);
            for (long int j = 0; j < N; j++) {
                if (ad_mat[current][j] > 0.0) { // اگر یال جهت‌دار وجود دارد
                    long double cost = 1.0 / ad_mat[current][j]; 
                    long double tentative_d = d[current] + cost;

                    // حالت اول: یک مسیر کاملاً کوتاه‌تر پیدا شد
                    if (tentative_d < d[j] - EPSILON) { 
                        d[j] = tentative_d;
                        sigma[j] = sigma[current];
                        
                        // پاک کردن پیشروهای قبلی و ثبت پیشرو جدید
                        for (long int k = 0; k < N; k++) P[j][k] = 0;
                        P[j][current] = 1;
                    } 
                    // حالت دوم: یک مسیر موازی با طول مساوی پیدا شد
                    else if (fabsl(tentative_d - d[j]) <= EPSILON) {
                        sigma[j] += sigma[current];
                        P[j][current] = 1; // اضافه کردن پیشرو جدید
                    }
                }
            }
        }

        // 3. مرحله Backward Pass (تخلیه پشته و محاسبه دلتا)
        while (stack_ptr > 0) {
            long int w = S[--stack_ptr]; // خارج کردن از پشته

            for (long int v = 0; v < N; v++) {
                if (P[w][v] == 1) { // اگر v پیشرو w است
                    // محاسبه سهم ترافیک و انتقال پاداش به عقب
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                }
            }
            // اگر گره فعلی خود مبدأ نبود، به امتیاز نهایی‌اش اضافه کن
            if (w != start) {
                BC[w] += delta[w];
            }
        }
    }
    
    //giving data to the property matrix
    for(long int e = 0; e < N;e++)
    {
        property[e] += BC[e];
    }

    // =========================================================
    // آزادسازی حافظه
    // =========================================================
    for (long int i = 0; i < N; i++) {
        free(P[i]);
    }
    free(P);
    free(d); free(sigma); free(delta); free(visited); free(S);
    free(BC);


    
}


void harmonic_closenness_in(long double **ad_mat,long double *property)
{
    double *d = (double *)calloc(N, sizeof(double));
    double *c_harm = (double *)calloc(N, sizeof(double));
    
    double *visited = (double *)calloc(N, sizeof(double));

    double **cost = (double **) calloc(N, sizeof(double *));
    for(long int i = 0; i < N; i++)
    {
        cost[i] = (double *) calloc(N, sizeof(double));
    }
    //calculating cost
    for(long int i = 0;i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            if(ad_mat[i][j] > 0)
            {
                cost[i][j] = (1.0/ad_mat[i][j]);
            }
            else
            {
                cost[i][j] = INFINITY;
            }
        }
    }

    //work
    for(long int s = 0;s < N;s++)
    {
        //making visted zero
        for(long int i = 0;i< N;i++)
        {
            visited[i] = 0;
        }
        //initializing d
        for(long int j = 0;j < N;j++)
        {
            d[j] = INFINITY;
        }   
        d[s] = 0;
        for(;;)
        {
            //calculatin min d
            double min = INFINITY;
            long int min_index;
            for(long int j = 0;j < N;j++)
            {
                if(d[j] < min && visited[j] == 0)
                {
                    min = d[j];
                    min_index = j;
                }
            }
            if(min == INFINITY)
            {
                break;
            }
    
            for(long int j = 0;j < N;j++)
            {
                if(ad_mat[j][min_index] > 0)
                {
                    double new_distance = d[min_index] + cost[j][min_index];//for in-Closeness
                    if(new_distance < d[j])
                    {
                        d[j] = new_distance;
                    }
                }
            }
            visited[min_index] = 1;
        }
        for(long int i = 0;i < N;i++)
        {
            if(s != i)
            {
                c_harm[s] += (1.0/d[i]);
            }
        }
    }
    for(long int e = 0; e < N;e++)
    {
        property[e] += c_harm[e];
    }

    for(long int i = 0; i < N; i++)
    {
        free(cost[i]);
    }
    
    free(c_harm);
    free(visited);
    free(d);
    free(cost);

    
}


void harmonic_closenness_out(long double **ad_mat,long double *property)
{
    double *d = (double *)calloc(N, sizeof(double));
    double *c_harm = (double *)calloc(N, sizeof(double));
    
    double *visited = (double *)calloc(N, sizeof(double));

    double **cost = (double **) calloc(N, sizeof(double *));
    for(long int i = 0; i < N; i++)
    {
        cost[i] = (double *) calloc(N, sizeof(double));
    }
    //calculating cost
    for(long int i = 0;i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            if(ad_mat[i][j] > 0)
            {
                cost[i][j] = (1.0/ad_mat[i][j]);
            }
            else
            {
                cost[i][j] = INFINITY;
            }
        }
    }

    //work
    for(long int s = 0;s < N;s++)
    {
        //making visted zero
        for(long int i = 0;i< N;i++)
        {
            visited[i] = 0;
        }
        //initializing d
        for(long int j = 0;j < N;j++)
        {
            d[j] = INFINITY;
        }   
        d[s] = 0;
        for(;;)
        {
            //calculatin min d
            double min = INFINITY;
            long int min_index;
            for(long int j = 0;j < N;j++)
            {
                if(d[j] < min && visited[j] == 0)
                {
                    min = d[j];
                    min_index = j;
                }
            }
            if(min == INFINITY)
            {
                break;
            }
    
            for(long int j = 0;j < N;j++)
            {
                if(ad_mat[min_index][j] > 0)
                {
                    double new_distance = d[min_index] + cost[min_index][j];//for Out-Closeness
                    if(new_distance < d[j])
                    {
                        d[j] = new_distance;
                    }
                }
            }
            visited[min_index] = 1;
        }
        for(long int i = 0;i < N;i++)
        {
            if(s != i)
            {
                c_harm[s] += (1.0/d[i]);
            }
        }
    }
    for(long int e = 0; e < N;e++)
    {
        property[e] += c_harm[e];
    }

    for(long int i = 0; i < N; i++)
    {
        free(cost[i]);
    }
    
    free(c_harm);
    free(visited);
    free(d);
    free(cost);

    
}


void leverage_in(long double **ad_mat,long double *property)
{
    long double *k_vec = calloc(N, sizeof(long double));
    double *c = calloc(N, sizeof(double));
    // محاسبه k_vec
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            k_vec[i] += ad_mat[j][i];
        }
    }

    for(int i = 0; i < N; i++)
    {
        if(k_vec[i] > 0)
        {
            for(int j = 0; j < N; j++)
            {
                if(ad_mat[j][i] > 0)
                {
                    c[i] += ((k_vec[i] - k_vec[j])/ (k_vec[i] + k_vec[j]));
                }
            }
            c[i] /= k_vec[i];
        }
    }
    for(long int e = 0; e < N;e++)
    {
        property[e] += c[e];
    }

    free(k_vec);
    free(c);
    
}


void leverage_out(long double **ad_mat,long double *property)
{
    long double *k_vec = calloc(N, sizeof(long double));
    double *c = calloc(N, sizeof(double));
    // محاسبه k_vec
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < N; j++)
        {
            k_vec[i] += ad_mat[i][j];
        }
    }

    for(int i = 0; i < N; i++)
    {
        if(k_vec[i] > 0)
        {
            for(int j = 0; j < N; j++)
            {
                if(ad_mat[i][j] > 0)
                {
                    c[i] += ((k_vec[i] - k_vec[j])/ (k_vec[i] + k_vec[j]));
                }
            }
            c[i] /= k_vec[i];
        }
    }
    for(long int e = 0; e < N;e++)
    {
        property[e] += c[e];
    }

    free(k_vec);
    free(c);
    
}


void PageRank_in(long double **ad_mat,long double *property)
{
    // 1. ساخت ماتریس GSL و کپی کردن داده‌های ad_mat درون آن
    gsl_matrix *M = gsl_matrix_alloc(N, N);
    for (long int i = 0; i < N; i++) {
        for (long int j = 0; j < N; j++) {
            gsl_matrix_set(M, i, j, (double)ad_mat[i][j]);
        }
    }

    long double *k_vec = calloc(N, sizeof(long double));


    //calculating k vectro
    for(int i = 0;i < N;i++)
    {
        for(int j = 0;j < N;j++)
        {
            k_vec[i]+=ad_mat[i][j];
        }
    }

    float d = 0.95;

    // 5. ساخت ماتریس (I - alpha * A)
    gsl_matrix *I_minus_d_M = gsl_matrix_alloc(N, N);
    double val;
    for (long int i = 0; i < N; i++) {
        for (long int j = 0; j < N; j++) {
            val = 0;
            if (k_vec[j] > 0.0) {
                val = -d * (gsl_matrix_get(M, j, i) / k_vec[j]);
            }
            else
            {
                val = -d * (1.0 / (double)N);
            }
            if (i == j) {
                val += 1.0; // اضافه کردن ماتریس همانی (Identity)
            }
            gsl_matrix_set(I_minus_d_M, i, j, val);
        }
    }

    // 6. محاسبه معکوس ماتریس (LU Decomposition)
    gsl_permutation *per = gsl_permutation_alloc(N);
    int signum;
    // تابع تجزیه LU، ماتریس ورودی را تغییر می‌دهد تا تجزیه را در آن ذخیره کند
    gsl_linalg_LU_decomp(I_minus_d_M, per, &signum); 
    
    gsl_matrix *inverse_mat = gsl_matrix_alloc(N, N);
    gsl_linalg_LU_invert(I_minus_d_M, per, inverse_mat);
    
    gsl_permutation_free(per);
    gsl_matrix_free(I_minus_d_M);

    // 7. محاسبه نهایی مرکزیت کاتز (PageRank Centrality)
    // با فرض اینکه بردار beta همگی 1 هستند، مرکزیت برابر است با مجموع عناصر هر سطر در ماتریس معکوس
    double *page_rank = (double *)calloc(N, sizeof(double));
    for (long int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (long int j = 0; j < N; j++) {
            row_sum += gsl_matrix_get(inverse_mat, i, j);//Broadcast Power or Out-Centrality
        }
        row_sum *= (double) ((1-d)/N);
        page_rank[i] = row_sum;
    }


    // 8. آزادسازی حافظه‌های ماتریسی
    gsl_matrix_free(inverse_mat);
    gsl_matrix_free(M);

    for(long int e = 0; e < N;e++)
    {
        property[e] += page_rank[e];
    }
    // آزادسازی ماتریس‌های GSL
    free(k_vec);
    free(page_rank);

    
}


void PageRank_out(long double **ad_mat,long double *property)
{
    // 1. ساخت ماتریس GSL و کپی کردن داده‌های ad_mat درون آن
    gsl_matrix *M = gsl_matrix_alloc(N, N);
    for (long int i = 0; i < N; i++) {
        for (long int j = 0; j < N; j++) {
            gsl_matrix_set(M, i, j, (double)ad_mat[i][j]);
        }
    }

    long double *k_vec = calloc(N, sizeof(long double));


    //calculating k vectro
    for(int i = 0;i < N;i++)
    {
        for(int j = 0;j < N;j++)
        {
            k_vec[i]+=ad_mat[j][i];
        }
    }

    float d = 0.95;

    // 5. ساخت ماتریس (I - alpha * A)
    gsl_matrix *I_minus_d_M = gsl_matrix_alloc(N, N);
    double val;
    for (long int i = 0; i < N; i++) {
        for (long int j = 0; j < N; j++) {
            val = 0;
            if (k_vec[j] > 0.0) {
                val = -d * (gsl_matrix_get(M, i, j) / k_vec[j]);
            }
            else
            {
                val = -d * (1.0 / (double)N);
            }
            if (i == j) {
                val += 1.0; // اضافه کردن ماتریس همانی (Identity)
            }
            gsl_matrix_set(I_minus_d_M, i, j, val);
        }
    }

    // 6. محاسبه معکوس ماتریس (LU Decomposition)
    gsl_permutation *per = gsl_permutation_alloc(N);
    int signum;
    // تابع تجزیه LU، ماتریس ورودی را تغییر می‌دهد تا تجزیه را در آن ذخیره کند
    gsl_linalg_LU_decomp(I_minus_d_M, per, &signum); 
    
    gsl_matrix *inverse_mat = gsl_matrix_alloc(N, N);
    gsl_linalg_LU_invert(I_minus_d_M, per, inverse_mat);
    
    gsl_permutation_free(per);
    gsl_matrix_free(I_minus_d_M);

    // 7. محاسبه نهایی مرکزیت کاتز (PageRank Centrality)
    // با فرض اینکه بردار beta همگی 1 هستند، مرکزیت برابر است با مجموع عناصر هر سطر در ماتریس معکوس
    double *page_rank = (double *)calloc(N, sizeof(double));
    for (long int i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (long int j = 0; j < N; j++) {
            row_sum += gsl_matrix_get(inverse_mat, i, j);//Broadcast Power or Out-Centrality
        }
        row_sum *= (double) ((1-d)/N);
        page_rank[i] = row_sum;
    }


    // 8. آزادسازی حافظه‌های ماتریسی
    gsl_matrix_free(inverse_mat);
    gsl_matrix_free(M);

    for(long int e = 0; e < N;e++)
    {
        property[e] += page_rank[e];
    }
    // آزادسازی ماتریس‌های GSL
    free(k_vec);
    free(page_rank);

    
}


typedef struct
{
    long int deg;
    long int *nbr;
} OutList;

/* Fast thread-local RNG */
static inline uint64_t splitmix64(uint64_t *state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline long int Random_Walk_directed_fast(const OutList *G, long int current, uint64_t *rng_state)
{
    if (G[current].deg == 0)
        return -1;

    uint64_t r = splitmix64(rng_state);
    return G[current].nbr[r % (uint64_t)G[current].deg];
}

void Random_Walk_Betweenness_directed(long double **ad_mat,long double *property,long int max_steps,long int MAX_WALKS)
{

    /* build out-degree and adjacency lists */
    long int *k = (long int *)calloc(N, sizeof(long int));
    OutList *G = (OutList *)calloc(N, sizeof(OutList));

    for (long int i = 0; i < N; i++)
    {
        for (long int j = 0; j < N; j++)
        {
            if (ad_mat[i][j] > 0)
                k[i]++;
        }
    }

    for (long int i = 0; i < N; i++)
    {
        G[i].deg = k[i];
        if (k[i] > 0)
            G[i].nbr = (long int *)malloc((size_t)k[i] * sizeof(long int));
        else
            G[i].nbr = NULL;
    }

    for (long int i = 0; i < N; i++)
    {
        long int idx = 0;
        for (long int j = 0; j < N; j++)
        {
            if (ad_mat[i][j] > 0)
                G[i].nbr[idx++] = j;
        }
    }

    /* parallel region */
    #pragma omp parallel
    {
        long int i, j;
        long int *visited = (long int *)calloc(N, sizeof(long int));
        long int *visited_standby = (long int *)calloc(N, sizeof(long int));
        long double *RWBC = (long double *)calloc(N, sizeof(long double));
        long double *RWBC_old = (long double *)calloc(N, sizeof(long double));
        long double *property_local = (long double *)calloc(N, sizeof(long double));

        uint64_t rng_state = 0x9e3779b97f4a7c15ULL ^
                             (uint64_t)time(NULL) ^
                             ((uint64_t)omp_get_thread_num() + 1ULL) * 0xD1B54A32D192ED03ULL;

        #pragma omp for schedule(dynamic)
        for (long s = 0; s < N; s++)
        {
            if (k[s] == 0)
                continue;

            for (long t = 0; t < N; t++)
            {
                if (s == t)
                    continue;

                long int successful_Walkers = 0;

                memset(visited, 0, (size_t)N * sizeof(long int));
                memset(RWBC, 0, (size_t)N * sizeof(long double));
                memset(RWBC_old, 0, (size_t)N * sizeof(long double));

                for (long int walks = 0; walks < MAX_WALKS; walks++)
                {
                    long int current = s;
                    memset(visited_standby, 0, (size_t)N * sizeof(long int));

                    for (long int step = 0; step < max_steps; step++)
                    {
                        current = Random_Walk_directed_fast(G, current, &rng_state);
                        if (current == -1)
                            break;

                        if (current != s)
                            visited_standby[current] += 1;

                        if (current == t)
                        {
                            successful_Walkers++;

                            visited_standby[current] = 0;
                            for (i = 0; i < N; i++)
                                visited[i] += visited_standby[i];

                            break;
                        }
                    }

                    if (successful_Walkers == 0)
                        continue;

                    for (i = 0; i < N; i++)
                        RWBC[i] = (successful_Walkers > 0) ? (long double)visited[i] / successful_Walkers : 0.0L;

                    if ((walks + 1) % 50 == 0)
                    {
                        long double error = 0.0L;

                        for (i = 0; i < N; i++)
                        {
                            if (fabsl(RWBC_old[i]) > 1e-18L)
                                error += fabsl(RWBC[i] - RWBC_old[i]) / fabsl(RWBC_old[i]);
                            else if (fabsl(RWBC[i]) > 1e-18L)
                                error += 1.0L;
                        }

                        for (i = 0; i < N; i++)
                            RWBC_old[i] = RWBC[i];

                        if (((long double)error/N) <= 1e-2L)
                            break;
                    }
                }

                if (successful_Walkers > 0)
                {
                    for (i = 0; i < N; i++)
                        property_local[i] += (long double)visited[i] / (long double)successful_Walkers;
                }
            }
        }

        #pragma omp critical
        {
            for (i = 0; i < N; i++)
                property[i] += property_local[i];
        }

        free(visited);
        free(visited_standby);
        free(RWBC);
        free(RWBC_old);
        free(property_local);
    }

    for (long int i = 0; i < N; i++)
        free(G[i].nbr);
    free(G);
    free(k);
}


void Harmonic_Random_Walk_Closeness_in(long double **ad_mat,long double *property,long int max_steps,long int MAX_WALKS)
{

    /* build out-degree and adjacency lists */
    long int *k = (long int *)calloc(N, sizeof(long int));
    OutList *G = (OutList *)calloc(N, sizeof(OutList));

    for (long int i = 0; i < N; i++)
    {
        for (long int j = 0; j < N; j++)
        {
            if (ad_mat[i][j] > 0)
                k[i]++;
        }
    }

    for (long int i = 0; i < N; i++)
    {
        G[i].deg = k[i];
        if (k[i] > 0)
            G[i].nbr = (long int *)malloc((size_t)k[i] * sizeof(long int));
        else
            G[i].nbr = NULL;
    }

    for (long int i = 0; i < N; i++)
    {
        long int idx = 0;
        for (long int j = 0; j < N; j++)
        {
            if (ad_mat[i][j] > 0)
                G[i].nbr[idx++] = j;
        }
    }

    /* parallel region */
    #pragma omp parallel
    {
        long int i, j;
        long double *property_local = (long double *)calloc(N, sizeof(long double));

        uint64_t rng_state = 0x9e3779b97f4a7c15ULL ^
                             (uint64_t)time(NULL) ^
                             ((uint64_t)omp_get_thread_num() + 1ULL) * 0xD1B54A32D192ED03ULL;

        #pragma omp for schedule(dynamic)
        for (long s = 0; s < N; s++)
        {
            

            for (long t = 0; t < N; t++)
            {
                if (k[t] == 0) continue;
                if (s == t)
                    continue;

                long int successful_Walkers = 0;
                long double pair_score_sum = 0.0L;
                long double pair_score_sum_old = 0.0L;
                
                


                for (long int walks = 0; walks < MAX_WALKS; walks++)
                {
                    long int current = t;

                    for (long int step = 0; step < max_steps; step++)
                    {
                        current = Random_Walk_directed_fast(G, current, &rng_state);
                        if (current == -1)
                            break;

                        if (current == s)
                        {
                            successful_Walkers++;

                            pair_score_sum += 1.0L / (long double)(step + 1);
                            break;
                        }
                    }

                    if (successful_Walkers == 0)
                        continue;

                    if ((walks + 1) % 50 == 0)
                    {
                        long double error = 0.0L;
                        long double pair_est = pair_score_sum / (long double)successful_Walkers;

                        if (fabsl(pair_score_sum_old) > 1e-18L)
                            error += fabsl(pair_est - pair_score_sum_old) / fabsl(pair_score_sum_old);
                        else if (fabsl(pair_est) > 1e-18L)
                            error += 1.0L;

                        
                        pair_score_sum_old = pair_est;

                        if (error <= 1e-2L)
                            break;
                    }
                }

                if (successful_Walkers > 0)
                {
                    property_local[s] += pair_score_sum / (long double)successful_Walkers;
                }
            }
        }

        #pragma omp critical
        {
            for (i = 0; i < N; i++)
                property[i] += property_local[i];
        }

        free(property_local);
    }

    for (long int i = 0; i < N; i++)
        free(G[i].nbr);
    free(G);
    free(k);
}


void Harmonic_Random_Walk_Closeness_out(long double **ad_mat,long double *property,long int max_steps,long int MAX_WALKS)
{

    /* build out-degree and adjacency lists */
    long int *k = (long int *)calloc(N, sizeof(long int));
    OutList *G = (OutList *)calloc(N, sizeof(OutList));

    for (long int i = 0; i < N; i++)
    {
        for (long int j = 0; j < N; j++)
        {
            if (ad_mat[i][j] > 0)
                k[i]++;
        }
    }

    for (long int i = 0; i < N; i++)
    {
        G[i].deg = k[i];
        if (k[i] > 0)
            G[i].nbr = (long int *)malloc((size_t)k[i] * sizeof(long int));
        else
            G[i].nbr = NULL;
    }

    for (long int i = 0; i < N; i++)
    {
        long int idx = 0;
        for (long int j = 0; j < N; j++)
        {
            if (ad_mat[i][j] > 0)
                G[i].nbr[idx++] = j;
        }
    }

    /* parallel region */
    #pragma omp parallel
    {
        long int i, j;
        long double *property_local = (long double *)calloc(N, sizeof(long double));

        uint64_t rng_state = 0x9e3779b97f4a7c15ULL ^
                             (uint64_t)time(NULL) ^
                             ((uint64_t)omp_get_thread_num() + 1ULL) * 0xD1B54A32D192ED03ULL;

        #pragma omp for schedule(dynamic)
        for (long s = 0; s < N; s++)
        {
            if (k[s] == 0)
                continue;

            for (long t = 0; t < N; t++)
            {
                if (s == t)
                    continue;

                long int successful_Walkers = 0;
                long double pair_score_sum = 0.0L;
                long double pair_score_sum_old = 0.0L;
                
                


                for (long int walks = 0; walks < MAX_WALKS; walks++)
                {
                    long int current = s;

                    for (long int step = 0; step < max_steps; step++)
                    {
                        current = Random_Walk_directed_fast(G, current, &rng_state);
                        if (current == -1)
                            break;

                        if (current == t)
                        {
                            successful_Walkers++;

                            pair_score_sum += 1.0L / (long double)(step + 1);
                            break;
                        }
                    }

                    if (successful_Walkers == 0)
                        continue;

                    if ((walks + 1) % 50 == 0)
                    {
                        long double error = 0.0L;
                        long double pair_est = pair_score_sum / (long double)successful_Walkers;

                        if (fabsl(pair_score_sum_old) > 1e-18L)
                            error += fabsl(pair_est - pair_score_sum_old) / fabsl(pair_score_sum_old);
                        else if (fabsl(pair_est) > 1e-18L)
                            error += 1.0L;

                        
                        pair_score_sum_old = pair_est;

                        if (error <= 1e-2L)
                            break;
                    }
                }

                if (successful_Walkers > 0)
                {
                    property_local[s] += pair_score_sum / (long double)successful_Walkers;
                }
            }
        }

        #pragma omp critical
        {
            for (i = 0; i < N; i++)
                property[i] += property_local[i];
        }

        free(property_local);
    }

    for (long int i = 0; i < N; i++)
        free(G[i].nbr);
    free(G);
    free(k);
}












/////////////////////////////////////////////////////////////////////////////////
//                             Network Equal                                   //
/////////////////////////////////////////////////////////////////////////////////














void Erdos_Renyi_directed_P(long double **ad_mat, long double p)
{
    // Initialize adjacency matrix with zeros
    for(long int i = 0; i < N; i++)
    {
        for(long int j = 0; j < N; j++)
        {
            ad_mat[i][j] = 0;
        }
    }

    // Randomly generate edges with probability p
    for(long int i = 0; i < N; i++)
    {
        for(long int j = 0; j < N; j++)
        {
            // No self-loops
            if(i != j && ad_mat[i][j] == 0.0)
            {
                long double r = (long double)rand() / RAND_MAX;

                if(r < p)
                {
                    ad_mat[i][j] = 1;
                }
            }
        }
    }
}


void Erdos_Renyi_directed_EQ(long double **ad_mat,long int E)
{
    //making zeros for ad_mat

    for(long int i = 0;i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            ad_mat[i][j] = 0;
        }
    }

    // ۵. جایگذاری تصادفی یال‌ها و اختصاص وزن‌ها
    long int edges_placed = 0;
    while (edges_placed < E) {
        long int u = rand() % N; // گره مبدأ تصادفی
        long int v = rand() % N; // گره مقصد تصادفی

        // شرایط: نباید طوقه (Self-loop) باشد و نباید قبلاً یالی در اینجا گذاشته باشیم
        if (u != v && ad_mat[u][v] == 0.0) {
            ad_mat[u][v] = 1;
            edges_placed++;
        }
    }
}


void Erdos_Renyi_EQ(long double **ad_mat,long int E)
{
    //making zeros for ad_mat

    for(long int i = 0;i < N;i++)
    {
        for(long int j = 0;j < N;j++)
        {
            ad_mat[i][j] = 0;
        }
    }

    // ۵. جایگذاری تصادفی یال‌ها و اختصاص وزن‌ها
    long int edges_placed = 0;
    while (edges_placed < E) {
        long int u = rand() % N; // گره مبدأ تصادفی
        long int v = rand() % N; // گره مقصد تصادفی

        // شرایط: نباید طوقه (Self-loop) باشد و نباید قبلاً یالی در اینجا گذاشته باشیم
        if (u != v && ad_mat[u][v] == 0.0) {
            ad_mat[u][v] = 1;
            ad_mat[v][u] = 1;
            edges_placed++;
        }
    }
}


void Bollobas_Borgs_Chayes_Riordan_EQ(long double **ad_mat,double alpha,double beta,double gamma,double delta_in,double delta_out)
{
    double *p_vec = (double *)calloc(N, sizeof(double));


    /// Initial network
    long int n = 2;
    ad_mat[0][1] = 1;

    ////////////////////////////////////////////////////////////////////////////////////////////correct
    // Growth phase

    double *d_out = (double *)calloc(N, sizeof(double)); 
    double *d_in = (double *)calloc(N, sizeof(double));
    for(long int i = 0;i < n;i++)
    {
        for(long int j = 0;j < n;j++)
        {
            if(ad_mat[i][j] > 0 && i!=j)
            {
                d_out[i]++;
            }
            if(ad_mat[j][i] > 0 && i!=j)
            {
                d_in[i]++;
            }
        }
    }
    do
    {
        double r = (float)rand() / RAND_MAX;
        if(r <= alpha)//out degree
        {
            double total_weight_in = 0.0;
            for(long int j = 0; j < n; j++) {
                total_weight_in += (d_in[j] + delta_in);
            }
            for(long int i = 0;i < n;i++)
            {
                p_vec[i] = ((d_in[i]+delta_in)/(total_weight_in));
            }
            double sum_temp = 0.0;
            for(long int j = 0; j < n; j++) sum_temp += p_vec[j];
            // Random number scaled to the remaining probability mass
            double r = ((double)rand() / RAND_MAX) * sum_temp;
            double cum = 0.0;
            for(long int j = 0; j < n; j++)
            {
                cum += p_vec[j];
                if(cum >= r)
                {
                    ad_mat[n][j] = 1;
                    d_out[n]++;
                    d_in[j]++;
                    break;
                }
            }
            n++;
        }
        else if(r > alpha && r <= (alpha + gamma))//in degree
        {
            double total_weight_out = 0.0;
            for(long int j = 0; j < n; j++) {
                total_weight_out += (d_out[j] + delta_out);
            }
            for(long int i = 0;i < n;i++)
            {
                p_vec[i] = ((d_out[i]+delta_out)/(total_weight_out));
            }
            double sum_temp = 0.0;
            for(long int j = 0; j < n; j++) sum_temp += p_vec[j];
            // Random number scaled to the remaining probability mass
            double r = ((double)rand() / RAND_MAX) * sum_temp;
            double cum = 0.0;
            for(long int j = 0; j < n; j++)
            {
                cum += p_vec[j];
                if(cum >= r)
                {
                    ad_mat[j][n] = 1;
                    d_in[n]++;
                    d_out[j]++;
                    break;
                }
            }
            n++;
        }
        else//add edge
        {
            double total_weight_out = 0.0;
            for(long int j = 0; j < n; j++) {
                total_weight_out += (d_out[j] + delta_out);
            }
            for(long int i = 0;i < n;i++)
            {
                if(d_out[i] < (n-1))
                {
                    p_vec[i] = ((d_out[i]+delta_out)/(total_weight_out));
                }
                else
                {
                    p_vec[i] = 0;
                }
                
            }
            long int start = -1;
            double sum_temp = 0.0;
            for(long int j = 0; j < n; j++) sum_temp += p_vec[j];
            if (sum_temp <= 0.0) continue; 
            // Random number scaled to the remaining probability mass
            double  cum;
            double r = ((double)rand() / RAND_MAX) * sum_temp;
            cum = 0.0;
            for(long int j = 0; j < n; j++)
            {
                cum += p_vec[j];
                if(cum >= r)
                {
                    start = j;
                    break;
                }
            }
            double total_weight_in = 0.0;
            for(long int j = 0; j < n; j++) {
                total_weight_in += (d_in[j] + delta_in);
            }
            for(long int i = 0;i < n;i++)
            {
                if(ad_mat[start][i] == 0 && start != i)
                {
                    p_vec[i] = ((d_in[i]+delta_in)/(total_weight_in));
                }
                else
                {
                    p_vec[i] = 0;
                }
            }
            sum_temp = 0.0;
            long int destination = -1;
            for(long int j = 0; j < n; j++) sum_temp += p_vec[j];
            if (sum_temp <= 0.0) continue; 
            // Random number scaled to the remaining probability mass
            
            r = ((double)rand() / RAND_MAX) * sum_temp;
            cum = 0.0;
            for(long int j = 0; j < n; j++)
            {
                cum += p_vec[j];
                if(cum >= r)
                {
                    destination = j;
                    break;
                }
            }
            
            ad_mat[start][destination] = 1;
            d_out[start]++;
            d_in[destination]++;
        }
    } while (n < N);
    
    
    free(d_in);
    free(d_out);
    free(p_vec);
        
    
}


void Bollobas_Borgs_Chayes_Riordan_undirect_EQ(long double **ad_mat,double alpha,double beta,double delta)
{
    double *p_vec = (double *)calloc(N, sizeof(double));


    /// Initial network
    long int n = 2;
    ad_mat[0][1] = 1;
    ad_mat[1][0] = 1;

    ////////////////////////////////////////////////////////////////////////////////////////////correct
    // Growth phase

    double *d = (double *)calloc(N, sizeof(double)); 
    for(long int i = 0;i < n;i++)
    {
        for(long int j = 0;j < n;j++)
        {
            if(ad_mat[i][j] > 0 && i!=j)
            {
                d[i]++;
            }
        }
    }
    do
    {
        double r = (float)rand() / RAND_MAX;
        if(r <= alpha)
        {
            double total_weight = 0.0;
            for(long int j = 0; j < n; j++) {
                total_weight += (d[j] + delta);
            }
            for(long int i = 0;i < n;i++)
            {
                p_vec[i] = ((d[i]+delta)/(total_weight));
            }
            double sum_temp = 0.0;
            for(long int j = 0; j < n; j++) sum_temp += p_vec[j];
            // Random number scaled to the remaining probability mass
            double r = ((double)rand() / RAND_MAX) * sum_temp;
            double cum = 0.0;
            for(long int j = 0; j < n; j++)
            {
                cum += p_vec[j];
                if(cum >= r)
                {
                    ad_mat[n][j] = 1;
                    ad_mat[j][n] = 1;
                    d[n]++;
                    d[j]++;
                    break;
                }
            }
            n++;
        }
        else//add edge
        {
            double total_weight = 0.0;
            for(long int j = 0; j < n; j++) {
                total_weight += (d[j] + delta);
            }
            for(long int i = 0;i < n;i++)
            {
                if(d[i] < (n-1))
                {
                    p_vec[i] = ((d[i]+delta)/(total_weight));
                }
                else
                {
                    p_vec[i] = 0;
                }
                
            }
            long int start = -1;
            double sum_temp = 0.0;
            for(long int j = 0; j < n; j++) sum_temp += p_vec[j];
            if (sum_temp <= 0.0) continue; 
            // Random number scaled to the remaining probability mass
            double  cum;
            double r = ((double)rand() / RAND_MAX) * sum_temp;
            cum = 0.0;
            for(long int j = 0; j < n; j++)
            {
                cum += p_vec[j];
                if(cum >= r)
                {
                    start = j;
                    break;
                }
            }

            double total_weight2 = 0.0;
            for(long int j = 0; j < n; j++) {
                total_weight2 += (d[j] + delta);
            }

            for(long int i = 0;i < n;i++)
            {
                if(ad_mat[start][i] == 0 && start != i)
                {
                    p_vec[i] = ((d[i]+delta)/(total_weight2));
                }
                else
                {
                    p_vec[i] = 0;
                }
            }
            sum_temp = 0.0;
            long int destination = -1;
            for(long int j = 0; j < n; j++) sum_temp += p_vec[j];
            if (sum_temp <= 0.0) continue; 
            // Random number scaled to the remaining probability mass
            
            r = ((double)rand() / RAND_MAX) * sum_temp;
            cum = 0.0;
            for(long int j = 0; j < n; j++)
            {
                cum += p_vec[j];
                if(cum >= r)
                {
                    destination = j;
                    break;
                }
            }
            
            ad_mat[start][destination] = 1;
            ad_mat[destination][start] = 1;
            d[start]++;
            d[destination]++;
        }
    } while (n < N);
    
    
    free(d);
    free(p_vec);
        
    
}



static int ring_distance(int i, int j, int L) {
    int d = abs(i - j);
    int alt = L - d;
    return (d < alt) ? d : alt;
}


static double edge_probability(int i, int j, int L, double p0, double beta) {
    int D = ring_distance(i, j, L);
    int Dmax = ceil(L / 2);  /* floor(L/2) */
    double d = 0.0;

    if (Dmax > 0) {
        d = (double)D / (double)Dmax;
    }

    /* p(d) = beta*p0 + (1-beta)*Theta(p0 - d) */
    if (d <= p0) {
        return beta * p0 + (1.0 - beta);
    } else {
        return beta * p0;
    }
}
void Watts_Strogatz_directed_EQ(long double **ad_mat,double p_ws,double Average_Degree_out)
{
        
    
    double p0 = ((double)Average_Degree_out / (double)(N - 1));

    
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if(i == j) continue;
            double p = edge_probability(i, j, N, p0, p_ws);
            if (rand() / ((double)RAND_MAX + 1.0) < p) {
                ad_mat[i][j] = 1;
            }
        }
    }
    
}


void Watts_Strogatz_EQ(long double **ad_mat,double p_ws,double Average_Degree)
{
    long int K = Average_Degree;
       // Connect each node to its K/2 neighbors on both sides.
       
    for (long int i = 0; i < N; i++) {
        for (long int j = 1; j <= K / 2; j++) {
            long int target = (i + j) % N;
            ad_mat[i][target] = 1;
            ad_mat[target][i] = 1;
        }
    }

    // Step 2: Rewiring Process
    // Iterate through nodes and only consider right-hand neighbors to ensure 
    // each edge is evaluated for rewiring exactly once without bias.
    for (long int i = 0; i < N; i++) {
        for (long int j = 1; j <= K / 2; j++) {
            
            // Evaluate probability p
            if (((float)rand() / RAND_MAX) <= p_ws) {
                
                long int orig_target = (i + j) % N;
                
                // 1. Remove the old edge
                ad_mat[i][orig_target] = 0;
                ad_mat[orig_target][i] = 0;
                
                // 2. Find a new random target node 'y'
                long int y;
                long int attempts = 0;
                do {
                    y = rand() % N;
                    attempts++;
                    // Failsafe to prevent infinite loops in highly dense graphs
                    if (attempts > N * 2) {
                        break; 
                    }
                } while (y == i || ad_mat[i][y] == 1);
                
                // 3. Establish the new edge (if a valid node was found)
                if (attempts <= N * 2) {
                    ad_mat[i][y] = 1;
                    ad_mat[y][i] = 1;
                } else {
                    // Restore the original edge if rewiring was impossible
                    ad_mat[i][orig_target] = 1;
                    ad_mat[orig_target][i] = 1;
                }
            }
        }
    }

}


void Configurational_model_directed_EQ(long double **ad_mat_real,long double **ad_mat_copy,long int E)
{
    for (long int i = 0; i < N; i++)
    {
        for (long int j = 0; j < N; j++)
        {
            ad_mat_copy[i][j] = ad_mat_real[i][j]; // Copy from real network
        }
    }
    
    // 2. Rewire to get a random graph with same degree sequence
    long int num_swaps = 100 * E; // Good mixing parameter
    
    //////////
    long int *edge_src = (long int *)malloc(E * sizeof(long int));
    long int *edge_dst = (long int *)malloc(E * sizeof(long int));
    long int edge_count = 0;
    
    // Build edge list from adjacency matrix
    for (long int i = 0; i < N; i++)
    {
        for (long int j = 0; j < N; j++)
        {
            if (ad_mat_copy[i][j] > 0)
            {
                edge_src[edge_count] = i;
                edge_dst[edge_count] = j;
                edge_count++;
            }
        }
    }
    
    // Perform edge swaps
    for (long int swap = 0; swap < num_swaps; swap++)
    {
        // Pick two distinct edges at random
        long int idx1 = rand() % E;
        long int idx2;
        do
        {
            idx2 = rand() % E;
        } while (idx1 == idx2);////////////
        
        int a = edge_src[idx1];
        int b = edge_dst[idx1];
        int c = edge_src[idx2];
        int d = edge_dst[idx2];
        
        // Check if swap is valid
        if (a == d || c == b) continue; // No self-loops
        if (a == c || b == d) continue; // Same nodes
        
        // Check if new edges already exist
        if (ad_mat_copy[a][d] > 0 || ad_mat_copy[c][b] > 0) continue;
        
        // Perform the swap in adjacency matrix
        long double x1;
        long double x2;
        
        x1 = ad_mat_copy[a][b];
        ad_mat_copy[a][b] = 0;
        x2 = ad_mat_copy[c][d];
        ad_mat_copy[c][d] = 0;
        ad_mat_copy[a][d] = x1;
        ad_mat_copy[c][b] = x2;
        
        // Update edge list
        edge_dst[idx1] = d;
        edge_dst[idx2] = b;
    }
}


#include <stdlib.h>

void Configurational_model_directed_percent(long double **ad_mat_real, long double **ad_mat_copy, long int E, long int percent)
{
    // 1. Copy from real network
    for (long int i = 0; i < N; i++)
    {
        for (long int j = 0; j < N; j++)
        {
            ad_mat_copy[i][j] = ad_mat_real[i][j]; 
        }
    }
    
    long int *edge_src = (long int *)malloc(E * sizeof(long int));
    long int *edge_dst = (long int *)malloc(E * sizeof(long int));
    long int edge_count = 0;
    
    // 2. Build edge list from adjacency matrix
    for (long int i = 0; i < N; i++)
    {
        for (long int j = 0; j < N; j++)
        {
            if (ad_mat_copy[i][j] > 0)
            {
                edge_src[edge_count] = i;
                edge_dst[edge_count] = j;
                edge_count++;
            }
        }
    }
    
    // 3. Create an active pool of available edges
    long int *available_edges = (long int *)malloc(E * sizeof(long int));
    for (long int i = 0; i < E; i++)
    {
        available_edges[i] = i;
    }
    long int num_available = E;
    
    long int successful_swaps = 0;
    long int max_possible_swaps = E / 2;
    long int attempts = 0;
    long int max_attempts = percent * max_possible_swaps;
    
    // 4. Perform edge swaps
    // Run as long as we have at least 2 unswapped edges remaining
    while (successful_swaps < max_possible_swaps && attempts < max_attempts && num_available >= 2)
    {
        attempts++;

        // Pick two distinct indices from the *available* pool
        long int r1 = rand() % num_available;
        long int r2;
        do
        {
            r2 = rand() % num_available;
        } while (r1 == r2);
        
        // Map the random selection to the actual edge indices
        long int idx1 = available_edges[r1];
        long int idx2 = available_edges[r2];
        
        int a = edge_src[idx1];
        int b = edge_dst[idx1];
        int c = edge_src[idx2];
        int d = edge_dst[idx2];
        
        // Check if swap is valid (if invalid, we just continue, they remain in the pool)
        if (a == d || c == b) continue; // No self-loops
        if (a == c || b == d) continue; // Same nodes
        if (ad_mat_copy[a][d] > 0 || ad_mat_copy[c][b] > 0) continue; // No multi-edges
        
        // Perform the swap in adjacency matrix
        long double x1 = ad_mat_copy[a][b];
        long double x2 = ad_mat_copy[c][d];
        
        ad_mat_copy[a][b] = 0;
        ad_mat_copy[c][d] = 0;
        ad_mat_copy[a][d] = x1;
        ad_mat_copy[c][b] = x2;
        
        // Update edge list
        edge_dst[idx1] = d;
        edge_dst[idx2] = b;
        
        // 5. SUCCESS: Remove these two edges from the available pool in O(1) time
        // We must remove the larger index first so it doesn't shift the position of the smaller one
        if (r1 < r2) 
        {
            long int temp = r1;
            r1 = r2;
            r2 = temp;
        }
        
        // Overwrite r1 with the last available edge, then shrink pool
        available_edges[r1] = available_edges[num_available - 1];
        num_available--;
        
        // Overwrite r2 with the new last available edge, then shrink pool
        available_edges[r2] = available_edges[num_available - 1];
        num_available--;
        
        successful_swaps++;
    }

    free(edge_src);
    free(edge_dst);
    free(available_edges);
}


void Configurational_model_undirected_EQ(long double **ad_mat_real, long double **ad_mat_copy, long int E)
{
    // 1. Copy from real network
    for (long int i = 0; i < N; i++)
    {
        for (long int j = 0; j < N; j++)
        {
            ad_mat_copy[i][j] = ad_mat_real[i][j]; 
        }
    }
    
    // 2. Rewire to get a random graph with same degree sequence
    long int num_swaps = 100 * E; // Good mixing parameter
    
    long int *edge_src = (long int *)malloc(E * sizeof(long int));
    long int *edge_dst = (long int *)malloc(E * sizeof(long int));
    long int edge_count = 0;
    
    // Build edge list from adjacency matrix (Upper triangle only for undirected)
    for (long int i = 0; i < N; i++)
    {
        for (long int j = i + 1; j < N; j++) // j = i + 1 prevents edge duplication
        {
            if (ad_mat_copy[i][j] > 0)
            {
                edge_src[edge_count] = i;
                edge_dst[edge_count] = j;
                edge_count++;
            }
        }
    }
    
    // Perform edge swaps
    for (long int swap = 0; swap < num_swaps; swap++)
    {
        // Pick two distinct edges at random
        long int idx1 = rand() % E;
        long int idx2;
        do
        {
            idx2 = rand() % E;
        } while (idx1 == idx2);
        
        long int a = edge_src[idx1];
        long int b = edge_dst[idx1];
        long int c = edge_src[idx2];
        long int d = edge_dst[idx2];
        
        // The two edges must be completely disjoint (no shared nodes)
        if (a == c || a == d || b == c || b == d) continue;
        
        long double w1 = ad_mat_copy[a][b];
        long double w2 = ad_mat_copy[c][d];

        // Randomly pick one of the two valid undirected rewiring orientations
        if (rand() % 2 == 0) 
        {
            // Option 1: Connect (a, d) and (b, c)
            if (ad_mat_copy[a][d] > 0 || ad_mat_copy[b][c] > 0) continue; // Multi-edge check

            // Remove old edges symmetrically
            ad_mat_copy[a][b] = 0; ad_mat_copy[b][a] = 0;
            ad_mat_copy[c][d] = 0; ad_mat_copy[d][c] = 0;

            // Add new edges symmetrically (maintaining original weights)
            ad_mat_copy[a][d] = w1; ad_mat_copy[d][a] = w1;
            ad_mat_copy[b][c] = w2; ad_mat_copy[c][b] = w2;

            // Update edge list
            edge_src[idx1] = a; edge_dst[idx1] = d;
            edge_src[idx2] = b; edge_dst[idx2] = c;
        } 
        else 
        {
            // Option 2: Connect (a, c) and (b, d)
            if (ad_mat_copy[a][c] > 0 || ad_mat_copy[b][d] > 0) continue; // Multi-edge check

            // Remove old edges symmetrically
            ad_mat_copy[a][b] = 0; ad_mat_copy[b][a] = 0;
            ad_mat_copy[c][d] = 0; ad_mat_copy[d][c] = 0;

            // Add new edges symmetrically (maintaining original weights)
            ad_mat_copy[a][c] = w1; ad_mat_copy[c][a] = w1;
            ad_mat_copy[b][d] = w2; ad_mat_copy[d][b] = w2;

            // Update edge list
            edge_src[idx1] = a; edge_dst[idx1] = c;
            edge_src[idx2] = b; edge_dst[idx2] = d;
        }
    }

    // Free dynamically allocated memory to prevent memory leaks
    free(edge_src);
    free(edge_dst);
}


void add_edge(long double **A, long int u, long int v)
{
    if (u == v) return;   /* no self-loops */
    A[u][v] = 1;
    A[v][u] = 1;
}
static long double rand01(void)
{
    return (long double)rand() / (long double)(RAND_MAX + 1.0);
}
void duplication_divergence_undirected(long double **A,int n_target, double p_keep, double p_parent)
{


    /* Seed graph: 2 nodes with one edge */
    add_edge(A, 0, 1);
    long int n_current = 2;

    while (n_current < n_target) {
        int edge = 0;
        long int parent;
        long int new_node;
        while(edge == 0)
        {
            
            parent = rand() % n_current;
            new_node = n_current;
    
            /* Copy edges from parent to new_node with probability p_keep */
            for (long int w = 0; w < n_current; w++) {
                if (w == parent) continue;
                if (A[parent][w] == 1) {
                    if (rand01() < p_keep) {
                        add_edge(A, new_node, w);
                        edge++;
                    }
                }
            }

            /* Optional direct link between parent and duplicate */
            if (rand01() < p_parent) {
                add_edge(A, new_node, parent);
                edge++;
            }
        }
        

       
        

        n_current++;
    }

}


void duplication_divergence_directed(long double **A,int n_target, double p_keep, double p_parent)
{


    /* Seed graph: 2 nodes with one edge */
    A[0][1] = 1.0L;
    long int n_current = 2;

    while (n_current < n_target) {
        int edge = 0;
        long int parent;
        long int new_node;
        while(edge == 0)
        {
            parent = rand() % n_current;
            new_node = n_current;
    
            /* Copy edges from parent to new_node with probability p_keep */
            for (long int w = 0; w < n_current; w++) {
                if (w == parent) continue;

                /* Copy outgoing edge parent -> w as new_node -> w */
                if (A[parent][w] == 1.0L && rand01() < p_keep) {
                    A[new_node][w] = 1.0L;
                    edge++;
                }
    
                /* Copy incoming edge w -> parent as w -> new_node */
                if (A[w][parent] == 1.0L && rand01() < p_keep) {
                    A[w][new_node] = 1.0L;
                    edge++;
                }
            }

            /* Optional direct link between parent and duplicate */
            if (rand01() < p_parent)
            {
                double r = rand01();
            
                if (r < 1.0/3.0)
                {
                    /* parent -> duplicate */
                    A[parent][new_node] = 1;
                }
                else if (r < 2.0/3.0)
                {
                    /* duplicate -> parent */
                    A[new_node][parent] = 1;
                }
                else
                {
                    /* bidirectional */
                    A[parent][new_node] = 1;
                    A[new_node][parent] = 1;
                }
                edge++;
            }
        }
        

       
        

        n_current++;
    }

}


void barabasi_albert_undirected(long double **ad_mat, long int N, long int m_0, long int m)
{
    if (N <= 0 || m_0 <= 0 || m <= 0 || m_0 > N || m > m_0) {
        fprintf(stderr, "Invalid parameters\n");
        return;
    }

    /* seed graph: complete graph on m_0 nodes */
    for (long int i = 0; i < m_0; i++) {
        for (long int j = i + 1; j < m_0; j++) {
            ad_mat[i][j] = 1.0L;
            ad_mat[j][i] = 1.0L;
        }
    }

    long double *k_vec = (long double *)calloc(N, sizeof(long double));
    long double *p_vec = (long double *)calloc(N, sizeof(long double));

    if (!k_vec || !p_vec) {
        fprintf(stderr, "Memory allocation failed\n");
        free(k_vec);
        free(p_vec);
        return;
    }

    for (long int i = 0; i < N - m_0; i++)
    {
        long int new_node = m_0 + i;

        /* degree vector */
        for (long int j = 0; j < new_node; j++) {
            k_vec[j] = 0.0L;
        }

        for (long int j = 0; j < new_node; j++) {
            for (long int k = 0; k < new_node; k++) {
                k_vec[j] += ad_mat[j][k];
            }
        }

        long double sum_k = 0.0L;
        for (long int j = 0; j < new_node; j++) {
            sum_k += k_vec[j];
        }

        for (long int j = 0; j < new_node; j++) {
            p_vec[j] = k_vec[j] / sum_k;
        }

        /* prevent duplicate targets */
        char *chosen = (char *)calloc(new_node, sizeof(char));
        if (!chosen) {
            fprintf(stderr, "Memory allocation failed\n");
            break;
        }

        int connected = 0;
        while (connected < m)
        {
            long double choose = (long double)rand() / ((long double)RAND_MAX + 1.0L);

            long double cum = 0.0L;
            for (long int k = 0; k < new_node; k++)
            {
                cum += p_vec[k];

                if (choose <= cum)
                {
                    if (!chosen[k] && ad_mat[new_node][k] == 0.0L)
                    {
                        ad_mat[new_node][k] = 1.0L;
                        ad_mat[k][new_node] = 1.0L;
                        chosen[k] = 1;
                        connected++;
                    }
                    break;
                }
            }
        }

        free(chosen);
    }

    free(k_vec);
    free(p_vec);
}








/////////////////////////////////////////////////////////////////////////////////
//                               properties                                    //
/////////////////////////////////////////////////////////////////////////////////












void Copy_Network(long double **ad_mat,long double ** ad_mat_real)
{
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
}


double shannon_entropy(long double *property)
{
    double min_property = property[0];
    for (long int i = 1; i < N; i++)
    {
        if (property[i] < min_property)
        {
            min_property = property[i];
        }
    }

    double sum_property = 0.0;
    double entropy = 0.0;
    // اگر مقادیر منفی داریم، همه را به اندازه کافی شیفت می‌دهیم تا مثبت شوند
    // اضافه کردن یک اپسیلون (مثل 1e-9) برای جلوگیری از صفر شدن مطلق مقادیر
    double shift_value = (min_property < 0) ? fabs(min_property) + 1e-9 : 0.0;

    for (long int i = 0; i < N; i++)
    {
        sum_property += (property[i] + shift_value);
    }

    if (sum_property > 0.0)
    {
        for (long int i = 0; i < N; i++)
        {
            double shifted_property = property[i] + shift_value;
            if (shifted_property > 0.0)
            { // هنوز این شرط لازم است تا از لگاریتم صفر جلوگیری کنیم
                double p = shifted_property / sum_property;
                entropy -= p * log(p);
            }
        }
    }
    return entropy;
}


void compute_histogram(long double *data, int data_size, long int *bins,double Number_Of_Bins, double min_val, double max_val) {// correct for positive numbers
    if (data_size == 0) return;

    // 1. Dynamically find the minimum and maximum values in the data

    


    // 3. Compute bin width (assuming global N represents the maximum number of bins)
    double bin_width = (max_val - min_val) / (double)Number_Of_Bins;

    // 4. Populate the histogram
    for (int i = 0; i < data_size; i++) {
        double val = (double)data[i];
        
        // Calculate the corresponding bin index
        int bin_index = (int)((val - min_val) / bin_width);
        
        // Handle edge cases (e.g., if a value equals max_val precisely)
        if (bin_index >= Number_Of_Bins) bin_index = Number_Of_Bins - 1;
        if (bin_index < 0) bin_index = 0;
        
        bins[bin_index]++;
    }
}


void empty_mat(long double **ad_mat)
{
    for(int i = 0;i < N;i++)
    {
        for(int j = 0;j < N;j++)
        {
            ad_mat[i][j] = 0;
        }
    }
}


void normalize(long double *property,long int n)
{
    double min_val = INFINITY;
    double max_val =  -INFINITY;
    for (int i = 0; i < n; i++) {
        if ((double)property[i] < min_val) min_val = (double)property[i];
        if ((double)property[i] > max_val) max_val = (double)property[i];
    }
    // Handle edge case where all nodes have the exact same centrality measure
    if (max_val == min_val) {
        max_val = min_val + 1.0; // Prevent division by zero
    }
    for (int i = 0; i < n; i++) {
        property[i] = ((property[i] - min_val) / (max_val - min_val));
    }
}


void empty_vec(long double * property )
{
    for(long int e = 0;e < N;e++)
    {
        property[e] = 0;
    }
}


void Pdf(long double *property,long int *bins,double Number_Of_Bins,long int n)
{
    for (long int k = 0; k < Number_Of_Bins; k++) { 
        // Average over iterations, then divide by N to get probability P(k)
        property[k] = bins[k] / ((long double)n);
        
    }
}


void is_zero_out_degree(long double **ad_mat)
{
    double s;
    for(long int i = 0;i < N;i++)
    {
        s=1;
        for(long int j = 0;j < N;j++)
        {
            if(ad_mat[i][j] > 0)
            {
                s = 0;
            }
        }
        if(s)
        {
            printf("node %ld has zero out degree\n",i);
        }
    }
    
}


typedef struct {
    int u; // Source (tail)
    int v; // Destination (head)
} DirectedEdge;

void generateLineDigraph(long double** A, int E, long double **A_L) {
    // Step 1: Extract directed edges
    DirectedEdge* edges = (DirectedEdge*)calloc(E , sizeof(DirectedEdge));
    long int edge_count = 0;
    
    for (long int i = 0; i < N; i++) {
        for (long int j = 0; j < N; j++) { // Scan the full matrix
            if (A[i][j]) {
                edges[edge_count].u = i;
                edges[edge_count].v = j;
                edge_count++;
            }
            if(i == j || A[i][j] == 0)
            {
                edges[edge_count].u = -1;
                edges[edge_count].v = -1;
            }
        }
    }


    // Step 2: Populate A_L_temp based on head-to-tail matching
    for (long int i = 0; i < E; i++) {
        for (long int j = 0; j < E; j++) { // Scan all pairs
            // If the destination of edge i is the source of edge j
            if (edges[i].v == edges[j].u && edges[i].v != (-1) && edges[j].u != (-1)) {
                A_L[i][j] = 1;
            }
        }
    }

    
    long int *k_in = (long int*)calloc(E , sizeof(long int));
    long int *k_out = (long int*)calloc(E , sizeof(long int));

    for (long int i = 0; i < E; i++)
    {
        for (long int j = 0; j < E; j++)
        {
            if(A_L[i][j] > 0)
            {
                k_out[i]++;
            }
            else if(A_L[j][i] > 0)
            {
                k_in[i]++;
            }
        }
    }

    for (long int i = 0; i < E; i++)
    {
        for (long int j = 0; j < E; j++)
        {
            if (k_in[i] == 0 && k_out[i] == 0)
            {
                A_L[i][j] = 0;
            }
        }
    }
    


    
    free(edges);
    free(k_in);
    free(k_out);
}


void generateLineGraph(long double** A, int E, long double **A_L) {
    // Step 1: Extract directed edges
    DirectedEdge* edges = (DirectedEdge*)calloc(E , sizeof(DirectedEdge));
    long int edge_count = 0;
    
    for (long int i = 0; i < N; i++) {
        for (long int j = (i+1); j < N; j++) { // Scan the full matrix
            if (A[i][j]) {
                edges[edge_count].u = i;
                edges[edge_count].v = j;
                edge_count++;
            }
            if(i == j || A[i][j] == 0)
            {
                edges[edge_count].u = -1;
                edges[edge_count].v = -1;
            }
        }
    }


    // Step 2: Populate A_L_temp based on head-to-tail matching
    for (long int i = 0; i < E; i++) {
        for (long int j = (i+1); j < E; j++) { // Scan all pairs
            // If the destination of edge i is the source of edge j
            if (edges[i].u != (-1) && edges[j].u != (-1)) {
                if(edges[i].u == edges[j].u || edges[i].u == edges[j].v || edges[i].v == edges[j].u || edges[i].v == edges[j].v)
                {
                    A_L[i][j] = 1;
                    A_L[j][i] = 1;
                }
            }
        }
    }

    
    long int *k = (long int*)calloc(E , sizeof(long int));

    for (long int i = 0; i < E; i++)
    {
        for (long int j = (i+1); j < E; j++)
        {
            if(A_L[i][j] > 0)
            {
                k[i]++;
                k[j]++;
            }
        }
    }

    for (long int i = 0; i < E; i++)
    {
        for (long int j = 0; j < E; j++)
        {
            if (k[i] == 0)
            {
                A_L[i][j] = 0;
                A_L[j][i] = 0;
            }
        }
    }
    


    
    free(edges);
    free(k);
}


#include <stdio.h>
#include <stdlib.h>

void VisualizeGraph(long double **ad_mat,
                    long int N,
                    const char *output_file,
                    const char *node_color,
                    const char *edge_color,
                    double node_size,
                    double edge_width,
                    int directed)
{
    char command[256];

    sprintf(command, "sfdp -Tpng -o %s", output_file);
    FILE *pipe = _popen(command, "w");

    if (!pipe)
    {
        perror("popen");
        return;
    }

    if (directed)
        fprintf(pipe, "digraph G {\n");
    else
        fprintf(pipe, "graph G {\n");

    fprintf(pipe,
    "graph [overlap=prism, pad=0.3, splines=false, "
    "outputorder=edgesfirst, dpi=300];\n");

    fprintf(pipe,
            "node [shape=circle, style=filled, fillcolor=\"%s\", "
            "width=%lf, height=%lf, fixedsize=true, fontsize=8];\n",
            node_color, node_size, node_size);

    fprintf(pipe,
            "edge [color=\"%s\", penwidth=%lf];\n",
            edge_color, edge_width);

    if (directed)
    {
        for (long int i = 0; i < N; i++)
        {
            for (long int j = 0; j < N; j++)
            {
                if (ad_mat[i][j] > 0)
                {
                    fprintf(pipe, "    %ld -> %ld;\n", i, j);
                }
            }
        }
    }
    else
    {
        for (long int i = 0; i < N; i++)
        {
            for (long int j = i + 1; j < N; j++)
            {
                if (ad_mat[i][j] > 0)
                {
                    fprintf(pipe, "    %ld -- %ld;\n", i, j);
                }
            }
        }
    }

    fprintf(pipe, "}\n");

    _pclose(pipe);
}


#define DEG_TOTAL 0
#define DEG_IN    1
#define DEG_OUT   2
static long int node_degree(long double **A, long int N, long int i, const char *mode)
{
    long int j;
    long int kin = 0, kout = 0;

    for (j = 0; j < N; j++) {
        if (A[j][i] != 0.0L) kin++;
        if (A[i][j] != 0.0L) kout++;
    }

    if (strcmp(mode,"in")==0)  return kin;
    if (strcmp(mode,"out")==0) return kout;
    return kin + kout;   /* DEG_TOTAL */
}
void DirectedClusteringCoefficient(long double **A, long int N,long double *C)
{
    long int i, j, k;
    long double sum, kin, kout, ktot, recip, denom, num;

    if (!A || !C || N <= 0) return;

    long double **B = (long double **) calloc(N, sizeof(long double *));
    for(int i = 0; i < N; i++)
    {
        B[i] = (long double *) calloc(N, sizeof(long double));
    }
     long double **B2 = (long double **) calloc(N, sizeof(long double *));
    for(int i = 0; i < N; i++)
    {
        B2[i] = (long double *) calloc(N, sizeof(long double));
    }

    /* Build B = A + A^T */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            B[i][j] = A[i][j] + A[j][i];
        }
    }

    /* Compute B2 = B * B */
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.0L;
            for (k = 0; k < N; k++) {
                sum += B[i][k] * B[k][j];
            }
            B2[i][j] = sum;
        }
    }


    for (i = 0; i < N; i++) {
        kin = 0.0L;
        kout = 0.0L;
        recip = 0.0L;

        /* in-degree, out-degree, and reciprocal links */
        for (j = 0; j < N; j++) {
            kin  += A[j][i];
            kout += A[i][j];
            recip += A[i][j] * A[j][i];
        }

        ktot = kin + kout;

        /*
            Denominator:
            2 * [k_i (k_i - 1) - 2 k_i^<->]
        */
        denom = 2.0L * (ktot * (ktot - 1.0L) - 2.0L * recip);

        /*
            Numerator: [(A + A^T)^3]_{ii}
            Using B2 = B^2, then diagonal of B^3 is sum_j B2[i][j] * B[j][i]
        */
        num = 0.0L;
        for (j = 0; j < N; j++) {
            num += B2[i][j] * B[j][i];
        }

        if (fabsl(denom) > 1e-18L) {
            C[i] = num / denom;
        } else {
            C[i] = 0.0L;
        }

    }


    for (i = 0; i < N; i++) free(B[i]);
    free(B);

    for (i = 0; i < N; i++) free(B2[i]);
    free(B2);
}


void NodePropertySpectrum(long double **A, long int N,long double *Ci, const char *mode,long double *Ck)
{
    long int i, k, kmax, deg;

    if (!A || !Ci || !Ck || N <= 0) return;

    if(strcmp(mode,"total")==0) kmax = 2*(N-1);
    else if(strcmp(mode,"in")==0 || strcmp(mode,"out")==0) kmax = N-1;
    else
    {
        fprintf(stderr,"Unknown mode: %s\n",mode);
        return;
    }

    
    long double *Nk = (long double *) calloc(kmax + 1, sizeof(long double ));
    long double *Ck_temp = (long double *) calloc(kmax + 1, sizeof(long double ));

    for (i = 0; i < N; i++) {
        deg = node_degree(A, N, i, mode);
        if (deg >= 0 && deg <= kmax) {
            Ck_temp[deg] += Ci[i];
            Nk[deg] += 1;
        }
    }

    for (k = 0; k <= kmax; k++) {
        if (Nk[k] > 0) Ck_temp[k] /= (long double)Nk[k];
        else Ck_temp[k] = -1.0L;
    }

    for(i = 0; i <= kmax; i++) Ck[i] = Ck_temp[i];

    free(Nk);
    free(Ck_temp);
}


void ClusteringCoefficientUndirected(long double **A, long int N, long double *C)
{
    long int i, j, k;
    long int ki;
    long int triangles;

    if (!A || !C || N <= 0) return;

    for (i = 0; i < N; i++) {
        ki = 0;
        triangles = 0;

        /* degree of node i */
        for (j = 0; j < N; j++) {
            if (A[i][j] != 0.0L) {
                ki++;
            }
        }

        /* if degree < 2, clustering is zero */
        if (ki < 2) {
            C[i] = 0.0L;
            continue;
        }

        /* count triangles through node i */
        for (j = 0; j < N; j++) {
            if (A[i][j] == 0.0L) continue;

            for (k = j + 1; k < N; k++) {
                if (A[i][k] == 0.0L) continue;

                /* j and k are both neighbors of i */
                if (A[j][k] != 0.0L || A[k][j] != 0.0L) {
                    triangles++;
                }
            }
        }

        C[i] = (2.0L * (long double)triangles) /
               ((long double)ki * (long double)(ki - 1));
    }
}


long double Average(long double *v,long int size)
{
    long double Average = 0;
    for(long int i = 0;i < size;i++)
    {
        Average += v[i];
    }
    Average /= (long double)size;
    return Average;
}


void ComputeKNN_Directed(long double **A,long int N,long double *knn,const char *neighbor_dir,const char *degree_dir)     /* "in" or "out" */
{
    long int i, j;

    long int *kin  = (long int *)calloc(N, sizeof(long int));
    long int *kout = (long int *)calloc(N, sizeof(long int));

    if(kin == NULL || kout == NULL)
    {
        printf("Memory allocation failed.\n");
        free(kin);
        free(kout);
        return;
    }

    /* Compute in-degree and out-degree */
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            if(A[i][j] != 0)
            {
                kout[i]++;   /* i -> j */
                kin[j]++;    /* j receives one incoming edge */
            }
        }
    }

    /* Compute k_nn for each node */
    for(i = 0; i < N; i++)
    {
        long double sum = 0.0;
        long int count = 0;

        for(j = 0; j < N; j++)
        {
            int is_neighbor = 0;
            long int deg = 0;

            /* choose neighbors: outgoing or incoming */
            if(strcmp(neighbor_dir, "out") == 0)
            {
                if(A[i][j] != 0)
                    is_neighbor = 1;
            }
            else if(strcmp(neighbor_dir, "in") == 0)
            {
                if(A[j][i] != 0)
                    is_neighbor = 1;
            }
            else
            {
                printf("Error: neighbor_dir must be \"in\" or \"out\".\n");
                free(kin);
                free(kout);
                return;
            }

            if(!is_neighbor)
                continue;

            /* choose which degree of the neighbor to use */
            if(strcmp(degree_dir, "out") == 0)
                deg = kout[j];
            else if(strcmp(degree_dir, "in") == 0)
                deg = kin[j];
            else
            {
                printf("Error: degree_dir must be \"in\" or \"out\".\n");
                free(kin);
                free(kout);
                return;
            }

            sum += (long double)deg;
            count++;
        }

        if(count == 0)
            knn[i] = 0.0;
        else
            knn[i] = sum / (long double)count;
    }

    free(kin);
    free(kout);
}


long double standard_deviation(long double *Network,long double average)
{
    long double std = 0;
    long double average_2 = 0;
    for(long int i = 0;i < N;i++)
    {
        average_2 += (Network[i] * Network[i]);
    }
    average_2/=N;

    std = sqrtl(average_2 - (average * average));
    return std;
}


long int max_core_number(long double **ad_mat, long int n)
{
    long int i, j;

    long int *deg = (long int *)calloc(n, sizeof(long int));
    int *removed = (int *)calloc(n, sizeof(int));

    if (deg == NULL || removed == NULL) {
        fprintf(stderr, "Memory allocation failed in max_core_number.\n");
        free(deg);
        free(removed);
        return -1;
    }

    /* Compute initial degrees */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && ad_mat[i][j] != 0.0L) {
                deg[i]++;
            }
        }
    }

    
    long int max_k = 0;

    /*
        Repeated peeling:
        We repeatedly remove nodes with smallest current degree.
        Since this is an O(n^2) matrix-based implementation,
        we search linearly for the minimum-degree alive node.
    */
    long int alive_count = n;

    while (alive_count > 0) {
        long int min_deg = n + 1;
        long int v = -1;

        /* find alive node with minimum degree */
        for (i = 0; i < n; i++) {
            if (!removed[i] && deg[i] < min_deg) {
                min_deg = deg[i];
                v = i;
            }
        }

        if (v == -1) {
            break;
        }

        if (min_deg > max_k) {
            max_k = min_deg;
        }

        removed[v] = 1;
        alive_count--;

        for (j = 0; j < n; j++) {
            if (!removed[j] && v != j && ad_mat[v][j] != 0.0L) {
                deg[j]--;
            }
        }
    }

    free(deg);
    free(removed);

    return max_k;
}


long int directed_max_k_core(long double **ad_mat, long int n, const char *mode)
{
    long int i, j;

    long int *deg = (long int *)calloc(n, sizeof(long int));
    int *removed = (int *)calloc(n, sizeof(int));

    if (deg == NULL || removed == NULL) {
        fprintf(stderr, "Memory allocation failed in directed_max_k_core.\n");
        free(deg);
        free(removed);
        return -1;
    }

    /* compute initial degrees according to the selected mode */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && ad_mat[i][j] != 0.0L) {
                if (strcmp(mode, "out") == 0 || strcmp(mode, "total") == 0) {
                    deg[i]++;   /* edge i -> j contributes to out-degree of i */
                }
                if (strcmp(mode, "in") == 0 || strcmp(mode, "total") == 0) {
                    deg[j]++;   /* same edge contributes to in-degree of j */
                }
            }
        }
    }

    long int alive = n;
    long int max_core = 0;

    /*
        Peeling process:
        repeatedly remove the alive node with minimum current degree.
        The largest minimum degree encountered is the maximum core number.
    */
    while (alive > 0) {
        long int min_deg = (2*(n + 1));
        long int v = -1;

        /* find alive node with minimum degree */
        for (i = 0; i < n; i++) {
            if (!removed[i] && deg[i] < min_deg) {
                min_deg = deg[i];
                v = i;
            }
        }

        if (v == -1) {
            break;
        }

        if (min_deg > max_core) {
            max_core = min_deg;
        }

        removed[v] = 1;
        alive--;

        /*
            Update neighbors:
            - "in":  removing v decreases in-degree of nodes j with v -> j
            - "out": removing v decreases out-degree of nodes j with j -> v
            - "total": both kinds of incident directed edges count
        */
        for (j = 0; j < n; j++) {
            if (removed[j] || j == v) {
                continue;
            }

            if ((strcmp(mode, "in") == 0 || strcmp(mode, "total") == 0) &&
                ad_mat[v][j] != 0.0L) {
                deg[j]--;
            }

            if ((strcmp(mode, "out") == 0 || strcmp(mode, "total") == 0) &&
                ad_mat[j][v] != 0.0L) {
                deg[j]--;
            }
        }
    }

    free(deg);
    free(removed);

    return max_core;
}


void make_directed(long double **ad_mat,long int n,long double p_bidirection)
{
    long double p = ((1-p_bidirection) / 2.0L);
    long double p1 = (2*p);
    for(long int i = 0; i < n;i++)
    {
        for(long int j = 0; j < n;j++)
        {
            if(ad_mat[i][j] > 0 && ad_mat[j][i] > 0)
            {
                if(rand01() < p)
                {
                    ad_mat[i][j] = 0;
                }
                if(p < rand01() < p1)
                {
                    ad_mat[j][i] = 0;
                }
                if(rand01() < p_bidirection)
                {
                    continue;
                }
            }
        }
    }
}


long double balance_index(long double **ad_mat, long int n)
{
    long int i, j;

    long int *in_deg  = (long int *)calloc(n, sizeof(long int));
    long int *out_deg = (long int *)calloc(n, sizeof(long int));

    if (in_deg == NULL || out_deg == NULL) {
        fprintf(stderr, "Memory allocation failed in directed_balance_index.\n");
        free(in_deg);
        free(out_deg);
        return -1.0L;
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i != j && ad_mat[i][j] != 0.0L) {
                out_deg[i]++;
                in_deg[j]++;
            }
        }
    }

    long double sum = 0.0L;
    for (i = 0; i < n; i++) {
        sum += fabsl((long double)(out_deg[i] - in_deg[i]));
    }

    free(in_deg);
    free(out_deg);

    return sum / (long double)n;
}


long double heterogeneity(long double **A, long int N, char mode[])
{
    long int i, j;
    long double k;
    long double sum_k = 0.0L;
    long double sum_k2 = 0.0L;

    for(i = 0; i < N; i++)
    {
        k = 0.0L;

        if(strcmp(mode, "out") == 0)
        {
            for(j = 0; j < N; j++)
                k += A[i][j];
        }
        else if(strcmp(mode, "in") == 0)
        {
            for(j = 0; j < N; j++)
                k += A[j][i];
        }
        else if(strcmp(mode, "total") == 0)
        {
            for(j = 0; j < N; j++)
                k += A[i][j] + A[j][i];
        }
        else
        {
            /* undirected */
            for(j = 0; j < N; j++)
                k += A[i][j];
        }

        sum_k += k;
        sum_k2 += k * k;
    }

    long double mean_k  = (long double)sum_k / N;
    long double mean_k2 = (long double)sum_k2 / N;

    if(mean_k == 0.0L)
        return 0.0L;

    return mean_k2 / (mean_k * mean_k);
}


void dfs_weak(long double **ad_mat, long int u, int *visited)
{
    visited[u] = 1;

    for(long int v = 0; v < N; v++)
    {
        if(!visited[v] &&
           (ad_mat[u][v] != 0.0L || ad_mat[v][u] != 0.0L))
        {
            dfs_weak(ad_mat, v, visited);
        }
    }
}

int WeaklyConnected(long double **ad_mat)
{
    int *visited = calloc(N, sizeof(int));

    if(visited == NULL)
        return 0;

    dfs_weak(ad_mat, 0, visited);

    for(long int i = 0; i < N; i++)
    {
        if(!visited[i])
        {
            free(visited);
            return 0;      // Graph is NOT weakly connected
        }
    }

    free(visited);
    return 1;              // Graph IS weakly connected
}