// To make cin & cout faster
ios_base::sync_with_stdio(0);
cin.tie(0);
cout.tie(0);

// Manual compare function for sorting
bool comp(pair<int, int> p1, pair<int, int> p2){
    // When p2 should come before p1 in the output
    if(p1.second < p2.second) // Don't use <=
        return 0;
    return 1;
}

//Backtracking
// Complete search - N queens
void Search(int y){
    if(y==n){
        cnt++;
        return;
    }
    for(int x=0; x<n; x++){
        if(r1[x]||r2[x+y]||r3[abs(x-y)])
            continue;
        r1[x] = r2[x+y] = r3[abs(x-y)] = 1;
        Search(y+1);
        r1[x] = r2[x+y] = r3[abs(x-y)] = 0;
    }
}

//Generating all subsets
void gen(int k, int n, int x) {
    if (k == n+1) {
        if(v.size()==x)
            sol.push_back(v);
    } else {
        gen(k+1, n, x);
        v.push_back (k);
        gen(k+1, n, x);
        v.pop_back();
    }
}

void Counting_Sort(int k){ // O(n+k)
    // k is maximum element
    int C[k+1];
    for(int i=0; i<=k; i++) C[i]=0;
    for(int j=0; j<n; j++)
        C[A[j]]++; // C[i] now contains number of elements equal to i
    for(int i=1; i<=k; i++)
        C[i]+=C[i-1]; // C[i] now contains number of elements less than or equal to i
    for(int j=n-1; j>=0; j--){
        B[C[A[j]]]=A[j];
        C[A[j]]--;
    }
}

// Standard binary search
void Binary_Search(int A[], int v){
    int low = 0;
    int high = A.length();
    while(low<=high){
        int mid=(low+high)/2;
        if(v=A[mid])
            return mid;
        if(v>A[mid])
            low=mid+1;
        else
            high=mid-1;
    }
    return null;
}

/********** Graph **********/

void BFS(int s){
    list<int> queue;
    visited[s] = 1;
    queue.push_back(s);
    while(!queue.empty()){
        s = queue.front();
        // process node
        queue.pop_front();
        for(auto i = adj[s].begin(); i != adj[s].end(); i++){
            if(!visited[*i]){
                visited[*i] = 1;
                queue.push_back(*i);
            }
        }
    }
}

int longestPath(int u){
    int dis[100001];
    memset(dis, -1, sizeof(dis));
    queue<int> q;
    q.push(u);
    dis[u] = 0;
    while (!q.empty()){
        int t = q.front();       q.pop();
        for (auto it = adj[t].begin(); it != adj[t].end(); it++)
        {
            int v = *it;
            if (dis[v] == -1)
            {
                q.push(v);
                dis[v] = dis[t] + 1;
            }
        }
    }
    int maxDis = 0;
    for (int i = 0; i < 100000; i++)
        if (dis[i] > maxDis)
            maxDis = dis[i];
    return maxDis;
}

void DFS(int s){
    visited[s]=1;
    //process node here
    for (int i=0; i<adj[s].size(); i++){
        if(!visited[adj[s][i]])
            DFS(adj[s][i]);
    }
}
//Dijkstra
int minDistance(int dist[], bool sptSet[]){
   int minn = INT_MAX, min_index;
   for (int v = 0; v < V; v++)
     if (sptSet[v] == false && dist[v] <= minn)
         minn = dist[v], min_index = v;
   return min_index;
}

void dijkstra(int graph[V][V], int src){
     int dist[V];     // The output array.  dist[i] will hold the shortest
                      // distance from src to i
     bool sptSet[V]; // sptSet[i] will true if vertex i is included in shortest
                     // path tree or shortest distance from src to i is finalized
     for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, sptSet[i] = false;
     dist[src] = 0;

     // Find shortest path for all vertices
     for (int count = 0; count < V-1; count++){
       int u = minDistance(dist, sptSet);
       sptSet[u] = true;
       // Update dist value of the adjacent vertices of the picked vertex.
       for (int v = 0; v < V; v++)
         // Update dist[v] only if is not in sptSet, there is an edge from
         // u to v, and total weight of path from src to  v through u is
         // smaller than current value of dist[v]
         if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u]+graph[u][v] < dist[v])
            dist[v] = dist[u] + graph[u][v];
     }
}


void dijkstra(){
	priority_queue< pair<double, int> > pq;
	pq.push(make_pair(1,1));
	while(!pq.empty()){
		pair<double, int> f = pq.top();
		pq.pop();
		double d = f.first;
		int u = f.second;
		vis[u] = 1;
		if(d < dist[u]) continue;
		for(int j = 0; j<adj[u].size(); j++){
			pair<int, double> v = adj[u][j];
			if(dist[u] * v.second > dist[v.first]){
				dist[v.first] = dist[u] * v.second;
				pq.push(make_pair(dist[v.first], v.first));
			}
		}
	}
}

// To find bridges in connected graph
void dfs(int v, int p = -1) {
    visited[v] = true;
    tin[v] = low[v] = timer++;
    for (int to : adj[v]) {
        if (to == p) continue;
        if (visited[to]) {
            low[v] = min(low[v], tin[to]);
        } else {
            dfs(to, v);
            low[v] = min(low[v], low[to]);
            if (low[to] > tin[v])
                IS_BRIDGE(v, to); // mark this pair as bridge
        }
    }
}

void find_bridges() {
    timer = 0;
    visited.assign(n, false);
    tin.assign(n, -1);
    low.assign(n, -1);
    for (int i = 0; i < n; ++i) {
        if (!visited[i])
            dfs(i);
    }
}

// Segment Tree
void build (int v, int tl, int tr) {
	if (tl == tr)
         tree[v] = b[tl];
	else {
		 int tm = (tl + tr) / 2;
         build (v*2, tl, tm);
         build (v*2+1, tm+1, tr);
         tree[v] = min (tree[v*2], tree[v*2+1]);
	}
}

int sum (int v, int tl, int tr, int l, int r) {
	if (l > r)
		return 0;
	if (l == tl && r == tr)
		return tree[v];
	int tm = (tl + tr) / 2;
	return min( sum (v*2, tl, tm, l, min(r,tm)),sum (v*2+1, tm+1, tr, max(l,tm+1), r));
}

void updateValue (int v, int tl, int tr, int pos, int new_val) {
	if (tl == tr)
        t[v] = new_val;
	else {
		int tm = (tl + tr) / 2;
		if (pos <= tm)
            update (v*2, tl, tm, pos, new_val);
		else
            update (v*2+1, tm+1, tr, pos, new_val);
        t[v] = t[v*2] + t[v*2+1];
	}
}

void updateRange(int si, int ss, int se, int us, int ue, int diff){
//si index, ss&se start and end of tree, us&ue start and end of range
    if (ss>se || ss>ue || se<us)
        return ;

    if (ss==se){
        tree[si]=diff;
        return;
    }

    int mid = (ss+se)/2;
    updateRange(si*2+1, ss, mid, us, ue, f);
    updateRange(si*2+2, mid+1, se, us, ue, f);

    tree[si] = min(tree[si*2+1], tree[si*2+2]);
}

//Lazy propagation
void propagate(int si, int ss, int se){
    if(lazy[si]==1 || lazy[si]==0){
        tree[si] = (se-ss+1) * lazy[si];
        if (ss != se){
            lazy[si*2]  = lazy[si];
            lazy[si*2+1]  = lazy[si];
        }
    }
    lazy[si] = -1;
}

int sum(int si, int ss, int se, int qs, int qe){
    if (lazy[si] != -1)
        propagate(si,ss,se);

    if (ss>se || ss>qe || se<qs)
        return 0;

    if (ss>=qs && se<=qe)
        return tree[si];

    int mid = (ss + se)/2;
    return sum(2*si, ss, mid, qs, qe) + sum(2*si+1, mid+1, se, qs, qe);
}

void updateRange(int si, int ss, int se, int us, int ue, int diff){
    if (lazy[si] != -1)
        propagate(si, ss, se);

    if (ss>se || ss>ue || se<us)
        return;

    if (ss>=us && se<=ue){
        lazy[si]=diff;
        propagate(si,ss,se);
        return;
    }
    int mid = (ss+se)/2;
    updateRange(si*2, ss, mid, us, ue, diff);
    updateRange(si*2+1, mid+1, se, us, ue, diff);

    tree[si] = tree[si*2] + tree[si*2+1];
}

// Another implementation
int RMQ(int ss, int se, int qs, int qe, int index)
{
    if (qs < 0 || qe > n-1 || qs > qe)
        return -1;
    if (qs <= ss && qe >= se)
        return st[index];
    if (se < qs || ss > qe)
        return 1e9;
    int mid = (ss+se)/2;
    return min(RMQ(ss, mid, qs, qe, 2*index+1),RMQ(mid+1, se, qs, qe, 2*index+2));
}

int constructST(int ss, int se, int si)
{
    if (ss == se)
    {
        st[si] = a[ss];
        return a[ss];
    }
    int mid = (ss+se)/2;
    st[si] =  min(constructST(ss, mid, si*2+1),constructST(mid+1, se, si*2+2));
    return st[si];
}

// Fenwick Tree or Binary Index Tree
int sum(int k){
    int s=0;
    while(k>=1){
        s+=b[k];
        k -= k&-k;
    }
    return s;
}

void add(int k, int x){
    while(k<=n){
        b[k]+=x;
        k += k&-k;
    }
}
int Find(int s){
    int index=-1;
    int low=0;
    int high = maxVal;
    for(int i=0; i<35; i++){
        int mid=(low+high)/2;
        if(get(mid)>=sum){
            index=mid;
            high=max(low,mid-1);
        }else{
            low=min(high,mid+1);
        }
    }
    return index;
}

// Better implementation
struct Fenwick{
    vector<int> tree;
    Fenwick(int n){
      tree.resize(n);
    }
    Fenwick(){}
    void add(int in, int val){
      in++;
      while(in < tree.size()){
        tree[in] += val;
        in += in & -in;
      }
    }
    ll get(int in){
      in++;
      ll res = 0;
      while(in){
        res += tree[in];
        in -= in & -in;
      }
      return res;
    }
    ll get(int l, int r){
      if(l == 0)return get(r);
      return get(r) - get(l - 1);
    }
    ll get_at(int in){
      return get(in, in);
    }
    int kth(int k){
        int cur = 0;
        int sum = 0;
        for(int i = 19;i >= 0;i--){
          int ncur = cur + (1 << i);
          if(ncur < tree.size() && sum + tree[ncur] < k){
            cur = ncur;
            sum += tree[ncur];
          }
        }
        return cur;
    }
};

// Union-Find Structure

for(int i=1; i<=n; i++){ // in main function
    k[i]=i;
    s[i]=1;
}
int Find(int x){
    return x==k[x]?x:k[x]=Find(k[x]);
}
bool same(int a, int b){
    return Find(a)==Find(b);
}
void Union(int a, int b){
    a = Find(a);
    b = Find(b);
    if(s[a]<s[b]) swap(a,b);
    s[a]+=s[b];
    k[b]=a;
}

//Prim Minimum Spanning Tree
int minKey(int key[], bool mstSet[]){
    int min = INT_MAX, min_index;
    for (int v = 0; v < V; v++)
        if (mstSet[v] == false && key[v] < min)
            min = key[v], min_index = v;
    return min_index;
}

int printMST(int parent[], int n, int graph[V][V]){
    cout<<"Edge   Weight\n";
    for (int i = 1; i < V; i++)
        printf("%d - %d    %d \n", parent[i], i, graph[i][parent[i]]);
}

void primMST(int graph[V][V]){
    int parent[V]; // Array to store constructed MST
    int key[V]; // Key values used to pick minimum weight edge in cut
    bool mstSet[V]; // To represent set of vertices not yet included in MST

    for (int i = 0; i < V; i++)
        key[i] = INT_MAX, mstSet[i] = false;

    key[0] = 0;
    parent[0] = -1;

    for (int cnt = 0; cnt < V - 1; cnt++){
        int u = minKey(key, mstSet);
        mstSet[u] = true;
        for (int v = 0; v < V; v++)
            if (graph[u][v] && mstSet[v] == false && graph[u][v] < key[v])
                parent[v] = u, key[v] = graph[u][v];
    }
    printMST(parent, V, graph);
}

//Sparse Table
int PreProcessing(void){
    for(int i=0; i<N; i++)
        sp[i][0]=arr[i];
    for(int j=1; 1<<j<=N; j++){
        for(int i=0; i<=N-(1<<j); i++){
            sp[i][j]=min(sp[i][j-1], sp[i+(1<<j-1)][j-1]);
        }
    }
}
int query(int L, int R){
    int K = log2(R-L+1);
    return min(sp[L][K], sp[R-(1<<K)+1][K]);
}

//Lowest Common Ancestor
void dfs(int in, int p){
    parent[in][0]=p;
    dep[in]=dep[p]+1;
    for(int i=0; i<adj[in].size(); i++){
        int child = adj[in][i];
        if(child!=p){
            (child,in);
        }
    }
}

void pre(void){
    for(int j=1; 1<<j<=N; j++){
        for(int i=0; i<=N; i++){
            if(parent[i][j-1]!=-1){
                parent[i][j]=parent[parent[i][j-1]][j-1];
            }
        }
    }
}

int query(int u, int v){
    if(dep[u]>dep[v])
        swap(u,v);
    for(int k=25; k>=0; k--){
        if((dep[v]-1<<k)>=dep[u]){
            v=parent[v][k];
        }
        if(u==v)
            return u;
    }
    for(int k=26; k>=0; k--){
        if(parent[u][k]!=-1 && parent[u][k]!=parent[v][k]){
            v=parent[v][k];
            u=parent[u][k];
        }
    }
    return parent[u][0];
}

//All Pair Distance Single Pair Distance - Floyd
int adj[100][100][100];
for(int k=1; i<N; k++)
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            adj[i][j][k]=min(adj[i][j][k-1],adj[i][k][k+1]+adj[k][j][k+1]);

for (int i = 0; i < V; i++)
    for (int j = 0; j < V; j++)
        p[i][j] = i; // initialize the parent matrix
for (int k = 0; k < V; k++)
    for (int i = 0; i < V; i++)
        for (int j = 0; j < V; j++) // this time, we need to use if statement
            if (AdjMat[i][k] + AdjMat[k][j] < AdjMat[i][j])
            {
                AdjMat[i][j] = AdjMat[i][k] + AdjMat[k][j];
                p[i][j] = p[k][j]; // update the parent matrix
            }
// when we need to print the shortest paths, we can call the method below:
void printPath(int i, int j)
{
    if (i != j)
        printPath(i, p[i][j]);
    printf(" %d", j);
}


/********* MATHS *********/

ll power(ll base, ll power) {
    ll result = 1;
    while(power > 0) {
        if(power&1)
            result = (result*base) % MOD;
        base = (base * base) % MOD;
        power = power / 2;
    }
    return result;
}

//GCD
// LCM = a*b/gcd(a,b)
int gcd(int a,int b){
  int c;
  while (a!=0) {
     c=a;
     a=b%a;
     b=c;
  }
  return b;
}

//Extended Euclid ax + by = g = gcd(x,y);
ll extended_euclid(ll a, ll b, ll &x, ll &y){
    if(a<0){
        ll r = extended_euclid(-a,b,x,y);
        x *= -1;
        return r;
    }
    if(b<0){
        ll r = extended_euclid(a,-b,x,y);
        y *= -1;
        return r;
    }
    if(b==0){
        x=1, y=0;
        return a;
    }
    ll g = extended_euclid(b, a%b, y, x);
    y -= a/b * x;
    return g;
}
// diophantine equation find pair (x,y) for ax+by=c if c%gcd(a,b)==0
ll dioph(ll a, ll b, ll c, ll &x, ll &y, ll &found){
    ll g = extended_euclid(a,b,x,y);
    if(found = c%g == 0){
        x *= c/g;
        y *= c/g;
    }
    return g;
}

//solves the equation ax = b (mod n)
vector<ll> ModularEquation(ll a, ll b, ll n){
    vector<ll> sols;
    ll x,y,g;
    g = extended_euclid(a, n, x, y);
    if(b%g != 0){
        return sols; // no solutions
    }
    x = ((x*b/g)%n+n)%n; //from LDE +ve mod
    for(int i=0; i<g; i++){
        sols.push_back((x+i*n/g)%n);
    }
    sort(sols.begin(),sols.end());
    return sols;
}

void primeFactors(int n){
    while (n%2 == 0){
        printf("%d ", 2);
        n = n/2;
    }
    for (int i = 3; i <= sqrt(n); i = i+2){
        while (n%i == 0){
            printf("%d ", i);
            n = n/i;
        }
    }
    if (n > 2)
        printf ("%d ", n);
}

// phi or totient function: number of positive integers
// less than n that are coprime to n
int phi(int n)
{
    float result = n;
    for (int p = 2; p * p <= n; ++p) {
        if (n % p == 0) {
            while (n % p == 0)
                n /= p;
            result *= (1.0 - (1.0 / (float)p));
        }
    }
    if (n > 1)
        result *= (1.0 - (1.0 / (float)n));
    return (int)result;
}

void phi_generator(void){
   for(int i=0; i<1000001; i++) primes[i]=1, phi[i]=1;
   for(int i=2; i<1000001; i++){
        if(primes[i]){
            phi[i]=i-1;
            for(int j=i*2; j<1000001; j+=i){
                primes[j]=0;
                int n=j, pow=1;
                while(!(n%i)) pow*=i, n/=i;
                phi[j] *= (pow/i)*(i-1);
            }
        }
   }
}


void Sieve(int n)
{
    // Create a boolean array "prime[0..n]" and initialize
    // all entries it as true. A value in prime[i] will
    // finally be false if i is Not a prime, else true.
    bool prime[n+1];
    memset(prime, true, sizeof(prime));

    for (int p=2; p*p<=n; p++)
    {
        // If prime[p] is not changed, then it is a prime
        if (prime[p] == true)
        {
            // Update all multiples of p
            for (int i=p*2; i<=n; i += p)
                prime[i] = false;
        }
    }

    // Print all prime numbers
    for (int p=2; p<=n; p++)
       if (prime[p])
          cout << p << " ";
}

int maxSubArraySum(int a[], int s){
   int max_so_far = a[0];
   int curr_max = a[0];
   for (int i = 1; i < s; i++){
        curr_max = max(a[i], curr_max+a[i]);
        max_so_far = max(max_so_far, curr_max);
   }
   return max_so_far;
}

//Modular arithmetic
long long modInverse(int a, int m){
    int g = gcd(a, m);
    if (g != 1)
        cout << "Inverse doesn't exist";
    else {
        // If a and m are relatively prime, then modulo inverse
        // is a^(m-2) mode m
        cout << "Modular multiplicative inverse is "
             << power(a, m-2, m);
    }
}

// To compute x^y under modulo m
int power(int x, unsigned int y, unsigned int m){
    if (y == 0)
        return 1;
    int p = (x, y/2, m) % m;
    p = (p * p) % m;

    return (y%2 == 0)? p : (x * p) % m;
}


ll inv(ll n, ll m) { // get n*? = 1 (mod m)
	ll la = 1, lb = 0, ra = 0, rb = 1;
	ll i = 0, t, mod = m;
	while(n%m) {
		if(!i)
			la -= n/m*ra, lb -= n/m*rb;
        else
			ra -= n/m*la, rb -= n/m*lb;
		i = !i;
		t = n, n = m, m = t%m;
	}
	return i ? (la%mod+mod)%mod : (ra%mod+mod)%mod;
}

/******** DP *********/

//Largest Increasing Subsequence
int numList[] = {5, 2, 7, 3, 4, 6};	// solution is finally set of 0s and 1s..pick or leave.
//			     0  1  0  1  1  1
int m = 7;
// called with LIS(0, m)
int LIS(int i, int prev)
{
	if(i == m)
		return 0;

	int choice1 = LIS(i+1, prev);	// LEAVE
	int choice2 = 0;

	if(numList[prev] <= numList[i])
		choice2 = LIS(i+1, i) + 1;

	return max(choice1, choice2);
}

// Another one
int LIS( int arr[], int n, int max_ref)
{
    if (n == 1)
        return 1;
    int res, max_ending_here = 1;
    for (int i = 1; i < n; i++){
        res = LIS(arr, i, max_ref);
        if (arr[i-1] < arr[n-1] && res + 1 > max_ending_here)
            max_ending_here = res + 1;
    }
    if (max_ref < max_ending_here)
       max_ref = max_ending_here;

    return max_ending_here;
}

// O(nlogn) LIS:
set<int> st;
set<int>::iterator it;
st.clear();
for(i=0; i<n; i++) {
  st.insert(array[i]);
  it=st.find(array[i]);
  it++;
  if(it!=st.end()) st.erase(it);
}
cout<<st.size()<<endl;

//Another one
int CeilIndex(vector<int> &v, int l, int r, int key) {
    while (r-l > 1) {
    int m = l + (r-l)/2;
    if (v[m] >= key) r = m;
    else l = m;
    }
    return r;
}

int LIS(vector<pair<ll,ll>> &v){
    if (v.size() == 0)
        return 0;
    vector<int> tail(v.size(), 0);
    int length = 1;
    tail[0] = v[0].second;
    for (size_t i = 1; i < v.size(); i++) {
        if (v[i].second < tail[0])
            tail[0] = v[i].second;
        else if (v[i].second > tail[length-1])
            tail[length++] = v[i].second;
        else
            tail[CeilIndex(tail, -1, length-1, v[i].second)] = v[i].second;
    }
    return length;
}

// knapsack size = 12
//	  0   1   0   0  1

const int MAX = 5;
int n = 5;
int weights[MAX] = {10, 4, 20, 5, 7};
int benfit[MAX] = {10, 15, 3, 1, 4};

// called with knapsack(0, intialWeight)
int knapsack(int i, int reminder)	// aka 0/1 knapsack
{
	if(i == n)
		return 0;

	int choice1 = knapsack(i+1, reminder);
	int choice2 = 0;

	if(reminder >= weights[i])
		choice2 = knapsack(i+1, reminder - weights[i]) + benfit[i];

	return max(choice1, choice2);
}

//put + or - to make the result %k==0
int fix(int a) {
	return (a % k + k) % k; // to process subtraction with mod
}

int tryAll2(int i, int mod) {
	int &ret = mem[i][mod];
	if (ret != -1)
		return ret;
	if (i == n)
		return ret = mod == 0;

	if (tryAll2(i + 1, fix(mod + v[i])) || tryAll2(i + 1, fix(mod - v[i])))
		return ret = 1;
	return ret = 0;

}

/********** BIG INTEGER ***********/

string longDivision(string number, int divisor){
    string ans;
    int idx = 0;
    int temp = number[idx] - '0';
    while (temp < divisor)
       temp = temp * 10 + (number[++idx] - '0');
    while (number.size() > idx){
        ans += (temp / divisor) + '0';
        temp = (temp % divisor) * 10 + number[++idx] - '0';
    }
    if (ans.length() == 0)
        return "0";
    return ans;
}

string multiplication(string s1, string s2){
    int a[101],b[101],ans[201]={0};
    int i,j=0,tmp;
    int l1 = s1.length();
    int l2 = s2.length();
    for(i = l1-1,j=0;i>=0;i--,j++){
        a[j] = s1[i]-'0';
    }
    for(i = l2-1,j=0;i>=0;i--,j++){
        b[j] = s2[i]-'0';
    }
    for(i = 0;i < l2;i++){
        for(j = 0;j < l1;j++){
            ans[i+j] += b[i]*a[j];
        }
    }
    for(i = 0;i < l1+l2;i++){
        tmp = ans[i]/10;
        ans[i] = ans[i]%10;
        ans[i+1] = ans[i+1] + tmp;
    }
    for(i = l1+l2; i>= 0;i--){
        if(ans[i] > 0)
            break;
    }
    string answer="";
    for(;i >= 0;i--){
        answer += ans[i]+'0';
    }
    return answer;
}
// SUBTRACT
string findDiff(string str1, string str2){
    string str = "";
    int n1 = str1.length(), n2 = str2.length();
    reverse(str1.begin(), str1.end());
    reverse(str2.begin(), str2.end());
    int carry = 0;
    for (int i=0; i<n2; i++) {
        int sub = ((str1[i]-'0')-(str2[i]-'0')-carry);
        if(sub<0){
            sub = sub + 10;
            carry = 1;
        }
        else carry = 0;

        str.push_back(sub + '0');
    }

    for (int i=n2; i<n1; i++) {
        int sub = ((str1[i]-'0') - carry);
        if(sub<0){
            sub = sub + 10;
            carry = 1;
        } else carry = 0;
        str.push_back(sub + '0');
    }
    reverse(str.begin(), str.end());

    return str;
}

/****** EVAlUATE ARITHMETIC PARENTHESES ******/

string to_string(int x){
    string s = "";
    while(x){
        s += x%10+'0';
        x/=10;
    }
    reverse(s.begin(), s.end());
    return s;
}
int to_int(string s){
    int x = 0, p = 1;
    reverse(s.begin(), s.end());
    for(int i=0; i<s.length(); i++){
        x += (s[i]-'0')*p;
        p *= 10;
    }
    return x;
}

int eval(string s){
    string tok = ""; // Do parantheses first
    for (int i = 0; i < s.length(); i++)
    {
        if (s[i] == '(')
        {
            int iter = 1;
            string token;
            i++;
            while (true)
            {
                if (s[i] == '(')
                {
                    iter++;
                } else if (s[i] == ')')
                {
                    iter--;
                    if (iter == 0)
                    {
                        i++;
                        break;
                    }
                }
                token += s[i];
                i++;
            }
            tok += to_string(eval(token));
        }
        tok += s[i];
    }

    for (int i = 0; i < tok.length(); i++)
    {
        if (tok[i] == '+')
        {
            return eval(tok.substr(0, i)) + eval(tok.substr(i+1, tok.length()-i-1));
        } else if (tok[i] == '-')
        {
            return eval(tok.substr(0, i)) - eval(tok.substr(i+1, tok.length()-i-1));
        }
    }

    for (int i = 0; i < tok.length(); i++)
    {
        if (tok[i] == '*')
        {
            return eval(tok.substr(0, i)) * eval(tok.substr(i+1, tok.length()-i-1));
        } else if (tok[i] == '/')
        {
            return eval(tok.substr(0, i)) / eval(tok.substr(i+1, tok.length()-i-1));
        }
    }

    return to_int(tok.c_str()); // Return the value...
}
