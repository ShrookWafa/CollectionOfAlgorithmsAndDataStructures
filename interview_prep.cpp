#include<iostream>

using namespace std;


/******** Linked List *********/

struct node
{
	int data;
	node *next;
	node *prev;
};

template <typename K> class list
{
    private:
    node *head, *tail;
    public:
    list(){
        head=NULL;
        tail=NULL;
    }
    void createnode(K value)
    {
        node *temp=new node;
        temp->data=value;
        temp->next=NULL;
        temp->prev=tail;
        if(head==NULL)
        {
            head=temp;
            tail=temp;
        }
        else
        {
            tail->next=temp;
            tail=temp;
        }
    }
    void display()
    {
        node *temp=new node;
        temp=head;
        while(temp!=NULL)
        {
            cout<<temp->data<<"\t";
            temp=temp->next;
        }
    }
    void display_backwards()
    {
        node *temp=new node;
        temp = tail;
        while(temp!=NULL){
            cout << temp->data << "\t";
            temp = temp->prev;
        }
    }
    void insert_start(K value)
    {
        node *temp=new node;
        temp->data=value;
        head->prev=temp;
        temp->next=head;
        temp->prev=NULL;
        head=temp;
    }
    void insert_position(int pos, K value)
    {
        node *pre=new node;
        node *cur=new node;
        node *temp=new node;
        cur=head;
        for(int i=1;i<pos;i++)
        {
            pre=cur;
            cur=cur->next;
        }
        temp->data=value;
        pre->next=temp;
        temp->prev=pre;
        temp->next=cur;
        cur->prev=temp;
    }
    void delete_first()
    {
        node *temp=new node;
        temp=head;
        head=head->next;
        head->prev=NULL;
        delete temp;
    }
    void delete_last()
    {
        node *current=new node;
        node *previous=new node;
        current=head;
        while(current->next!=NULL)
        {
            previous=current;
            current=current->next;
        }
        tail=previous;
        previous->next=NULL;
        delete current;
    }
    void delete_position(int pos)
    {
        node *current=new node;
        node *previous=new node;
        current=head;
        for(int i=1;i<pos;i++)
        {
            previous=current;
            current=current->next;
        }
        node *temp = current;
        previous->next=current->next;
        current->next->prev=previous;
        delete temp;
    }
};

int main()
{
	list<int> obj;
	obj.createnode(25);
	obj.createnode(50);
	obj.createnode(90);
	obj.createnode(40);
	obj.display_backwards();
	obj.createnode(55);
	obj.display_backwards();
	obj.insert_start(50);
	obj.display_backwards();
	obj.insert_position(5,60);
	obj.display_backwards();
	obj.delete_first();
	obj.display_backwards();
	obj.delete_last();
	obj.display_backwards();
	obj.delete_position(4);
	obj.display_backwards();

	return 0;
}


/******** Sorting Algorithms **********/

void Merge(int a[], int low, int mid, int high){
    int n = mid-low+1, m = high-mid;
    int L[n], R[m];
    for(int i=0; i<n; i++){
        L[i] = a[i+low];
    }
    for(int i=0; i<m; i++){
        R[i] = a[i+mid+1];
    }
    int i=0, j=0, k=low;
    while(i<n && j<m){
        if(L[i]<=R[j]){
            a[k]=L[i];
            i++;
        }else{
            a[k]=R[j];
            j++;
        }
        k++;
    }
    while(i<n){
        a[k]=L[i];
        i++;
        k++;
    }
    while(j<m){
        a[k]=R[j];
        j++;
        k++;
    }
}

void MergeSort(int a[], int low, int high){  // O(nlogn)
    if(low<high){
        int mid = (low+high)/2;
        MergeSort(a, low, mid);
        MergeSort(a, mid+1, high);
        Merge(a,low,mid,high);
    }
}

/////////

void heapify(int &arr[], int n, int i) {
    int largest = i;
    int l = 2*i + 1;
    int r = 2*i + 2;

    if (l < n && arr[l] > arr[largest])
        largest = l;

    if (r < n && arr[r] > arr[largest])
        largest = r;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSort(int &arr[], int n) {   // O(nlogn) unstable
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    for (int i=n-1; i>=0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

/////////

int getMax(int arr[], int n)
{
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}

void countSort(int arr[], int n, int exp){
    int output[n];
    int i, count[10] = {0};

    for (i = 0; i < n; i++)
        count[ (arr[i]/exp)%10 ]++;

    for (i = 1; i < 10; i++)
        count[i] += count[i - 1];

    for (i = n - 1; i >= 0; i--) {
        output[count[ (arr[i]/exp)%10 ] - 1] = arr[i];
        count[ (arr[i]/exp)%10 ]--;
    }

    for (i = 0; i < n; i++)
        arr[i] = output[i];
}

void radixsort(int arr[], int n) {
    int m = getMax(arr, n);
    for (int exp = 1; m/exp > 0; exp *= 10)
        countSort(arr, n, exp);
}

///////////

int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = (low - 1);
    for (int j = low; j <= high- 1; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

void quickSort(int arr[], int low, int high){ // Best: O(nlogn) Worst O(n^2)
    if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

/////////

void insertionSort(int arr[], int n){ // O(n^2)
    int i, key, j;
    for (i = 1; i < n; i++){
        key = arr[i];
        j = i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

/////////

void BubbleSort(int arr[], int n) // O(n^2)
{
   int i, j;
   for (i = 0; i < n-1; i++)
       for (j = 0; j < n-i-1; j++)
           if (arr[j] > arr[j+1])
              swap(&arr[j], &arr[j+1]);
}

/******** Binary Search Tree **********/

struct node{
    int val;
    node *left;
    node *right;
};

class TreeSort{
public:
    node *root;
    TreeSort(){
        root = NULL;
    }
    node* getNode(int val){
        node* temp = new node;
        temp->val = val;
        temp->left = NULL;
        temp->right = NULL;
        return temp;
    }
    void Insert(int val, node *r){
        if(root == NULL){
            root = getNode(val);
            return;
        }
        if(val > r->val){
            if(r->right==NULL) r->right = getNode(val);
            else Insert(val, r->right);
        }else if(val < r->val){
            if(r->left==NULL) r->left = getNode(val);
            else Insert(val, r->left);
        }
    }
    node* getMin(node* r){
        node* temp = r;
        while(temp->left!=NULL){
            temp = temp->left;
        }
        return temp;
    }
    node* Delete(int val, node *r){
        if(r==NULL) return r;
        if(val < r->val){
            r->left = Delete(val, r->left);
        }else if(val > r->val){
            r->right = Delete(val, r->right);
        }else{
            if(r->left==NULL){
                node* temp = r->right;
                free(r);
                return temp;
            }
            if(r->right==NULL){
                node* temp = r->left;
                free(r);
                return temp;
            }
            node* temp = getMin(r->right);
            r->val = temp->val;
            r->right = Delete(temp->val, r->right);
        }
        return r;
    }
    void display_sorted(node *r){
        if(r!=NULL){
            display_sorted(r->left);
            cout << r->val << " ";
            display_sorted(r->right);
        }
    }
};

int main(){
    int a[] = {5, 2, -1, 10, 7};
    TreeSort TS;
    //TS.root = NULL;
    for(int i=0; i<5; i++) TS.Insert(a[i],TS.root);
    TS.Delete(5, TS.root);
    TS.display_sorted(TS.root);

    return 0;
}

/*********** Hash Table ***********/

class Hash
{
    int BUCKET;    // No. of buckets
    // Pointer to an array containing buckets
    list<int> *table;
public:
    Hash(int V);  // Constructor

    void insertItem(int x);
    void deleteItem(int key);
    int hashFunction(int x) {
        return (x % BUCKET);
    }

    void displayHash();
};

Hash::Hash(int b)
{
    this->BUCKET = b;
    table = new list<int>[BUCKET];
}

void Hash::insertItem(int key)
{
    int index = hashFunction(key);
    table[index].push_back(key);
}

void Hash::deleteItem(int key)
{
    // get the hash index of key
    int index = hashFunction(key);

    // find the key in (inex)th list
    list <int> :: iterator i;
    for (i = table[index].begin(); i != table[index].end(); i++) {
        if (*i == key)
            break;
    }

    // if key is found in hash table, remove it
    if (i != table[index].end())
      table[index].erase(i);
}

void Hash::displayHash() {
    for (int i = 0; i < BUCKET; i++) {
        cout << i;
        for (auto x : table[i])
            cout << " --> " << x;
        cout << endl;
    }
}

int main(){
    int a[] = {15, 11, 27, 8, 12};
    int n = sizeof(a)/sizeof(a[0]);
    Hash h(7);   // 7 is count of buckets in
    for (int i = 0; i < n; i++)
      h.insertItem(a[i]);

    h.deleteItem(12);

    h.displayHash();

    return 0;
}
