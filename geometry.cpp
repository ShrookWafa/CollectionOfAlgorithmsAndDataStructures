#include <bits/stdc++.h>

using namespace std;

struct Point{
    double x, y;
    Point(double x, double y):x(x), y(y){}

    Point():x(0), y(0){}
    Point& operator=(const Point& o){
        x = o.x;
        y = o.y;
        return *this;
    }
    Point& operator+=(const Point& o){
        x += o.x;
        y += o.y;
        return *this;
    }
    Point& operator-=(const Point& o){
        x -= o.x;
        y -= o.y;
        return *this;
    }
    Point& operator*=(double fact){
        x -= fact;
        y -= fact;
        return *this;
    }
    Point& operator/=(double fact){
        x /= fact;
        y /= fact;
        return *this;
    }
};
Point any;

/* rotate in 3d
Point rotate(const Point &p, double an1, double an2){
    return Point(cos(an2)*p.x + sin(an2)*sin(an1)*p.y - sin(an2)*cos(an1)*p.z,
                 cos(an1)*p.y + sin(an1)*p.z,
                 sin(an2)*p.x + cos(an2)*(-sin(an1))*p.y + cos(an2)*cos(an1)*p.z);
}
*/
Point operator-(const Point &a){
    return Point(-a.x, -a.y);
}
Point operator+(const Point &a, const Point &b){
    return Point(a.x+b.x, a.y+b.y);
}
Point operator-(const Point &a, const Point &b){
    return Point(a.x-b.x, a.y-b.y);
}
double operator*(const Point &a, const Point &b){
    return a.x*b.x + a.y*b.y;
}
double operator^(const Point &a, const Point &b){
    return a.x*b.y - a.y*b.x;
}
Point operator*(const double factor, const Point & p){
    return Point(factor * p.x, factor * p.y);
}
Point operator*(const Point & p, const double factor){
    return Point(factor * p.x, factor * p.y);
}
bool operator==(const Point & a, const Point & b){
    return a.x == b.x && a.y == b.y;
}
bool operator!=(const Point & a, const Point & b){
    return a.x != b.x || a.y != b.y;
}
double angle(const Point& p){
    return atan2(p.y, p.x);
}
double angle(const Point& a, const Point& b){
    return atan2(a^b, a*b);
}
Point rotate(const Point &p, double an){
    return Point(p.x * sin(an) - p.y * cos(an), p.x * cos(an) + p.y * sin(an));
}
Point rotate(const Point &p, double an, Point& around){
    return rotate(p - around, an) + around;
}

void rotate_around_x(Point& p, double th){
    p.x = p.x;
    double x, y;
    x = p.y, y = p.z;
    p.y = x * cos(th) - y * sin(th);
    p.z = x * sin(th) + y * cos(th);
}

void rotate_around_z(Point& p, double th){
    p.z = p.z;
    double x, y;
    x = p.x, y = p.y;
    p.x = x * cos(th) - y * sin(th);
    p.y = x * sin(th) + y * cos(th);
}

double abs(const Point &p){
    return p*p;
}
double norm(const Point &p){
    return sqrt(p*p);
}
Point perp(const Point &p){
    return Point(-p.y, p.x);
}
Point bisector(const Point &a, const Point &b){
    return a * norm(b) + b * norm(a);
}
double proj(const Point &a, const Point &b){
    return a * b / norm(b);
}
bool PointInPolygon(Point point, vector<Point>points) {
  int i, j, nvert = points.size();
  bool c = false;
  for(i = 0, j = nvert - 1; i < nvert; j = i++) {
    if( ( (points[i].y >= point.y ) != (points[j].y >= point.y) ) &&
        (point.x <= (points[j].x - points[i].x) * (point.y - points[i].y) / (points[j].y - points[i].y) + points[i].x)
      )
      c = !c;
  }
  return c;
}
// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

// Another struct for convex hull
struct Point {
	double x, y;

	bool operator <(const Point &p) const {
		return x < p.x || (x == p.x && y < p.y);
	}
};

double cross(const Point &O, const Point &A, const Point &B)
	return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);

vector<Point> convex_hull(vector<Point> P){
	int n = P.size(), k = 0;
	if (n <= 3) return P;
	vector<Point> H(2*n);
	sort(P.begin(), P.end());
	for (int i = 0; i < n; ++i) {
		while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}
	for (int i = n-1, t = k+1; i > 0; --i) {
		while (k >= t && cross(H[k-2], H[k-1], P[i-1]) <= 0) k--;
		H[k++] = P[i-1];
	}
	H.resize(k-1);
	return H;
}

//--------------------------------------lines------------------------------------------------


struct Line{
    Point a, ab;
    Line(const Point &a, const Point &b):a(a), ab(b-a){}
    Line():a(), ab(){}

    Point b(){
        return a + ab;
    }

};


bool online(const Line& l, Point& p){
    return ((p - l.a) ^ l.ab) == 0;
}


double dist(const Line& l, Point& p){
        return abs(((p-l.a)^l.ab)/norm(l.ab));
}

bool inter(const Line& s1, const Line &s2, Point& res){
    if((s1.ab ^ s2.ab) == 0)return 0; // parallel
    double t = ((s2.a - s1.a) ^ s2.ab) / (s1.ab ^ s2.ab);
    res = s1.a + s1.ab * t;
    return 1;
}

double proj(const Point &p, const Line &l, Point &res){
    res = l.a;
    res += l.ab *  ( ((p - l.a) * l.ab) / abs(l.ab) );
}
void reflection(const Point &p, const Line &l, Point &res){
    proj(p, l, res);
	res = 2 * res - p;
}

//-----------------------------segment----------------------------------



struct Segment{
    Point a, ab;
    Segment(const Point &a, const Point &b):a(a), ab(b-a){}
    Segment():a(), ab(){}

    Point b () const {
        return a + ab;
    }
};


bool onsegment(const Segment& r, const Point& p){
    return ((p - r.a) ^ r.ab) == 0 && ((p - r.a) * (p - r.b())) <= 0;
}



double dist(const Segment& r, const Point& p){
        if((p - r.a) * (r.ab) <= 0)return norm(p - r.a);
        if((p - r.b()) * (-r.ab) <= 0)return norm(p - r.b());

        return abs(((p-r.a)^r.ab)/norm(r.ab));
}



bool bet(const Segment &s1, const Segment &s2, const Point &p){
    return (dist(s1, p) + dist(s2, p) == dist(s2, s1.a));
}

bool inter(const Segment& s1, const Segment &s2, Point& res = any){
    if((s1.ab ^ s2.ab) == 0)return 0; // parallel
    double t1 = ((s2.a - s1.a) ^ s2.ab) / (s1.ab ^ s2.ab);
    double t2 = ((s1.a - s2.a) ^ s1.ab) / (s2.ab ^ s1.ab);
    if(t1 < 0 || t1 > 1 || t2 < 0 || t2 > 1)return 0;// does not intersect
    res = s1.a + s1.ab * t1;
    return 1;
}


int main()
{
//    freopen("in.txt", "r", stdin);
//    freopen("out.txt", "w", stdout);
    int tc;
    cin >> tc;
    while(tc--){
        double x1, y1, x2, y2, xl, yt, xd, yb;
        cin >> x1 >> y1 >> x2 >> y2 >> xl >> yt >> xd >> yb;
        Segment r1(Point(xl, yt), Point(xl, yb)),
        r2(Point(xl, yt), Point(xd, yt)),
        r3(Point(xd, yb), Point(xl, yb)),
        r4(Point(xd, yb), Point(xd, yt));

        Segment s (Point(x1, y1), Point(x2, y2));
//        cout << inter(s, r1) << inter(s, r2) << inter(s, r3) << inter(s, r4) << endl;
        if(inter(s, r1)
        || inter(s, r2)
        || inter(s, r3)
        || inter(s, r4)
        || bet(r1, r4, s.a)
        || bet(r1, r4, s.b())
        ){
            cout << "T\n";
        }else{
            cout << "F\n";
        }
    }
    return 0;
}
