#include "halfedge.h"

#include <set>
#include <vector>
#include <string>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <spdlog/spdlog.h>

using Eigen::Matrix3f;
using Eigen::Matrix4f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using std::optional;
using std::set;
using std::size_t;
using std::string;
using std::unordered_map;
using std::vector;

HalfedgeMesh::EdgeRecord::EdgeRecord(unordered_map<Vertex*, Matrix4f>& vertex_quadrics, Edge* e) :
    edge(e)
{
    (void)vertex_quadrics;
    optimal_pos = Vector3f(0.0f, 0.0f, 0.0f);
    cost        = 0.0f;
}

bool operator<(const HalfedgeMesh::EdgeRecord& a, const HalfedgeMesh::EdgeRecord& b)
{
    if (a.cost == b.cost) {
        // Sort by edge id if cost are the same
        return a.edge->id < b.edge->id;
    }
    return a.cost < b.cost;
}

optional<Edge*> HalfedgeMesh::flip_edge(Edge* e)
{
    if (e == nullptr) {
        return std::nullopt;
    }
    logger->trace("---start flip_edge {}---" , e->id);
    if (e->on_boundary()) {
        logger->trace("boundary edge");
        return std::nullopt;
    }
    //日志打印辅助lambda函数
    auto printh = [&] (Halfedge* h)
    {
        logger->trace("h {} next {} prev {} inv {} v {} e {} f {}" ,
                      h->id , h->next->id , h->prev->id , h->inv->id , h->from->id , h->edge->id , h->face->id);
    };
    auto printface = [&] (Face* f)
    {
        logger->trace("face {}" , f->id);
        printh(f->halfedge);
        printh(f->halfedge->next);
        printh(f->halfedge->prev);
    };
    auto is_triangle = [] (Halfedge* h)
    {
        if (h == nullptr) {
            return false;
        }
        return h->next->next == h->prev;
    };
    //所需边
    auto h12 = e->halfedge; //h43
    auto h21 = h12->inv; //h34
    //检测两边是否为三角面
    if (!is_triangle(h12) || !is_triangle(h21)) {
        logger->trace("flip requires two triangle faces");
        return std::nullopt;
    }
    auto h31 = h12->prev;
    auto h42 = h21->prev;
    auto h14 = h21->next;
    auto h23 = h12->next;
    //所需点
    auto v1 = h12->from;
    auto v2 = h21->from;
    auto v3 = h31->from;
    auto v4 = h42->from;
    //所需面
    auto f1 = h12->face;
    auto f2 = h21->face;
    //开始日志
    logger->trace("v1 {} v2 {} v3 {} v4 {}" , v1->id , v2->id , v3->id , v4->id);
    logger->trace("h12 {} h23 {} h31 {} " , h12->id , h23->id , h31->id);
    logger->trace("h21 {} h14 {} h42 {} " , h21->id , h14->id , h42->id);
    printface(f1);
    printface(f2);
    //v3和v4如果本来就相连，不执行翻转
    Halfedge* p = v3->halfedge;
    do {
        if (p->inv->from == v4) {
            logger->trace("flip would create a duplicate edge");
            return std::nullopt;
        }
        p = p->inv->next;
    } while (p != v3->halfedge);
    //面修复
    do{
        f1->halfedge = h12;
        f2->halfedge = h21;
    }while (false);
    //修正边
    do{
        h12->set_neighbors(h31 , h14 , h21 , v4 , e , f1);
        h21->set_neighbors(h42 , h23 , h12 , v3 , e , f2);
        h23->next = h21;
        h23->prev = h42;
        h23->face = f2;
        h31->next = h14;
        h31->prev = h12;
        h14->next = h12;
        h14->prev = h31;
        h14->face = f1;
        h42->next = h23;
        h42->prev = h21;
    }while (false);
    //修正点
    do{
        v1->halfedge = h14;
        v2->halfedge = h23;
        v3->halfedge = h31;
        v4->halfedge = h42;
    }while (false);
    //结束日志
    printface(f1);
    printface(f2);
    validate();
    logger->trace("---end---");
    return e;
}

optional<Vertex*> HalfedgeMesh::split_edge(Edge* edge)
{
    if (edge == nullptr) {
        return std::nullopt;
    }
    logger->trace("---split edge {}---" , edge->id);
    //若是边界则修改其方向
    const bool boundary = edge->on_boundary();
    if (edge->halfedge->is_boundary()) {
        edge->halfedge = edge->halfedge->inv;
    }
    //检侧三角面
    auto is_triangle = [] (Halfedge* h)
    {
        if (h == nullptr) {
            return false;
        }
        return h->next->next == h->prev;
    };
    if (!is_triangle(edge->halfedge) || (!boundary && !is_triangle(edge->halfedge->inv))) {
        logger->trace("no a real adjacent triangle");
        return std::nullopt;
    }
    //声明存储数组
    vector<vector<Halfedge*>> h(5 , vector<Halfedge*>(5));
    vector<Edge*> e(4);
    vector<Vertex*> v(5);
    vector<Face*> f(4);
    //日志打印辅助lambda函数
    auto printv = [&] (const int i)
    {
        logger->trace("vertex{} {}" , i , v[i]->id);
    };
    auto printe = [&] (const int i)
    {
        logger->trace("edge{} {}" , i , e[i]->id);
    };
    auto printh = [&] (const int i , const int j)
    {
        const auto he = h[i][j];
        logger->trace("h{}{} {} next {} prev {} inv {} v {} e {} f {}" ,
                      i , j , he->id , he->next->id , he->prev->id , he->inv->id , he->from->id , he->edge->id ,
                      he->face->id);
    };
    auto printface = [&] (const int i)
    {
        const auto face = f[i];
        logger->trace("face{} {} h {} {} {}" ,
                      i , face->id , face->halfedge->id , face->halfedge->next->id , face->halfedge->prev->id);
    };
    //半边
    do {
        h[0][1] = edge->halfedge;
        h[1][0] = h[0][1]->inv;
        h[1][2] = h[0][1]->next;
        h[2][0] = h[0][1]->prev;
        h[0][4] = h[0][1];
        h[1][4] = h[1][0];
        printh(0 , 1);
        printh(1 , 0);
        printh(1 , 2);
        printh(2 , 0);
        h[4][1] = new_halfedge();
        h[4][0] = new_halfedge();
        h[2][4] = new_halfedge();
        h[4][2] = new_halfedge();
        if (!boundary) {
            h[0][3] = h[1][0]->next;
            h[3][1] = h[1][0]->prev;
            printh(0 , 3);
            printh(3 , 1);
            h[3][4] = new_halfedge();
            h[4][3] = new_halfedge();
        }
    } while (false);
    //从新点出发的边
    do {
        e[0] = edge;
        e[1] = new_edge();
        e[2] = new_edge();
        e[2]->is_new = true;
        printe(0);
        printe(1);
        printe(2);
        if (!boundary) {
            e[3] = new_edge();
            e[3]->is_new = true;
            printe(3);
        }
    } while (false);
    //点
    do {
        v[0] = h[0][1]->from;
        v[1] = h[1][0]->from;
        v[2] = h[2][0]->from;
        for (int i = 0 ; i < 3 ; ++i) {
            printv(i);
        }
        if (!boundary) {
            v[3] = h[3][1]->from;
            printv(3);
        }
        v[4] = new_vertex();
        v[4]->is_new = true;
    } while (false);
    //面
    do {
        f[0] = h[0][1]->face;
        f[1] = h[1][0]->face;
        printface(0);
        printface(1);
        f[2] = new_face();
        if (!boundary) {
            f[3] = new_face();
        }
    } while (false);
    //连接半边
    do {
        h[0][4]->set_neighbors(h[4][2] , h[2][0] , h[4][0] , v[0] , e[0] , f[0]);
        h[4][2]->set_neighbors(h[2][0] , h[0][4] , h[2][4] , v[4] , e[2] , f[0]);
        h[2][0]->set_neighbors(h[0][4] , h[4][2] , h[2][0]->inv , v[2] , h[2][0]->edge , f[0]);
        h[4][1]->set_neighbors(h[1][2] , h[2][4] , h[1][4] , v[4] , e[1] , f[2]);
        h[1][2]->set_neighbors(h[2][4] , h[4][1] , h[1][2]->inv , v[1] , h[1][2]->edge , f[2]);
        h[2][4]->set_neighbors(h[4][1] , h[1][2] , h[4][2] , v[2] , e[2] , f[2]);
        printh(0 , 4);
        printh(4 , 2);
        printh(2 , 0);
        printh(4 , 1);
        printh(1 , 2);
        printh(2 , 4);
        if (!boundary) {
            h[4][0]->set_neighbors(h[0][3] , h[3][4] , h[0][4] , v[4] , e[0] , f[1]);
            h[0][3]->set_neighbors(h[3][4] , h[4][0] , h[0][3]->inv , v[0] , h[0][3]->edge , f[1]);
            h[3][4]->set_neighbors(h[4][0] , h[0][3] , h[4][3] , v[3] , e[3] , f[1]);
            h[1][4]->set_neighbors(h[4][3] , h[3][1] , h[4][1] , v[1] , e[1] , f[3]);
            h[4][3]->set_neighbors(h[3][1] , h[1][4] , h[3][4] , v[4] , e[3] , f[3]);
            h[3][1]->set_neighbors(h[1][4] , h[4][3] , h[3][1]->inv , v[3] , h[3][1]->edge , f[3]);
            printh(4 , 0);
            printh(0 , 3);
            printh(3 , 4);
            printh(1 , 4);
            printh(4 , 3);
            printh(3 , 1);
        }
        else {
            h[4][0]->set_neighbors(h[1][0]->next , h[1][4] , h[0][4] , v[4] , e[0] , f[1]);
            h[1][4]->set_neighbors(h[4][0] , h[1][4]->prev , h[4][1] , v[1] , e[1] , f[1]);
            printh(4,0);
            printh(1,4);
        }
    } while (false);
    //连接边
    do {
        e[0]->halfedge = h[0][4];
        e[1]->halfedge = h[1][4];
        e[2]->halfedge = h[2][4];
        printe(0);
        printe(1);
        printe(2);
        if (!boundary) {
            e[3]->halfedge = h[3][4];
            printe(3);
        }
    } while (false);
    //连接点
    do {
        v[0]->halfedge = h[0][4];
        v[1]->halfedge = h[1][4];
        v[2]->halfedge = h[2][4];
        v[4]->halfedge = h[4][0];
        for (int i = 0 ; i < 3 ; ++i) {
            printv(i);
        }
        if (!boundary) {
            v[3]->halfedge = h[3][4];
            printv(3);
        }
        v[4]->pos = (v[0]->pos + v[1]->pos) / 2;
    } while (false);
    //连接面
    do {
        f[0]->halfedge = h[0][4];
        f[1]->halfedge = h[4][0];
        f[2]->halfedge = h[4][1];
        for (int i = 0 ; i < 3 ; ++i) {
            printface(i);
        }
        if (!boundary) {
            f[3]->halfedge = h[1][4];
            printface(3);
        }
    } while (false);
    validate();
    logger->trace("---end---");
    return v[4];
}

optional<Vertex*> HalfedgeMesh::collapse_edge(Edge* edge)
{
    if (edge == nullptr) {
        return std::nullopt;
    }
    logger->trace("---collapse edge {} begin---" , edge->id);
    //若是边界则执行方向修正
    const bool boundary = edge->on_boundary();
    if (boundary && edge->halfedge->is_boundary()) {
        edge->halfedge = edge->halfedge->inv;
    }
    //两边三角面检测
    auto is_triangle = [] (Halfedge* h)
    {
        if (h == nullptr) {
            return false;
        }
        return h->next->next == h->prev;
    };
    if (!is_triangle(edge->halfedge) || (!boundary && !is_triangle(edge->halfedge->inv))) {
        logger->trace("collapse requires two triangle faces");
        return std::nullopt;
    }
    vector<vector<Halfedge*>> h(5 , vector<Halfedge*>(5));
    vector<vector<Edge*>> e(5 , vector<Edge*>(5));
    vector<Vertex*> v(5);
    vector<Face*> f(2);
    //半边
    do {
        h[0][1] = edge->halfedge;
        h[1][0] = h[0][1]->inv;
        h[1][2] = h[0][1]->next;
        h[2][1] = h[1][2]->inv;
        h[2][0] = h[0][1]->prev;
        h[0][2] = h[2][0]->inv;
        h[2][4] = h[2][1];
        h[4][2] = h[0][2];
        if (!boundary) {
            h[0][3] = h[1][0]->next;
            h[3][0] = h[0][3]->inv;
            h[3][1] = h[1][0]->prev;
            h[1][3] = h[3][1]->inv;
            h[4][3] = h[1][3];
            h[3][4] = h[3][0];
        }
    } while (false);
    //边
    do {
        e[0][1] = edge;
        e[0][2] = h[0][2]->edge;
        e[1][2] = h[1][2]->edge;
        e[2][4] = e[0][2];
        if (!boundary) {
            e[0][3] = h[0][3]->edge;
            e[1][3] = h[1][3]->edge;
            e[3][4] = e[0][3];
        }
    } while (false);
    //点
    do {
        v[0] = h[0][1]->from;
        v[1] = h[1][2]->from;
        v[2] = h[2][0]->from;
        if (!boundary) {
            v[3] = h[3][0]->from;
        }
        v[4] = v[0];
    } while (false);
    //检查v0，v1的领域交集是否只有v2,v3
    do {
        std::unordered_set<Vertex*> set;
        auto p = v[0]->halfedge;
        do {
            set.insert(p->inv->from);
            p = p->prev->inv;
        } while (p != v[0]->halfedge);
        p = v[1]->halfedge;
        int num{0};
        do {
            if (set.contains(p->inv->from)) {
                ++num;
            }
            p = p->prev->inv;
        } while (p != v[1]->halfedge);
        if (num != (boundary ? 1 : 2)) {
            return std::nullopt;
        }
    } while (false);
    //面
    do {
        f[0] = h[0][1]->face;
        if (!boundary) {
            f[1] = h[1][0]->face;
        }
    } while (false);
    //修改半边
    do {
        Halfedge* p = h[1][0];
        do {
            p->from = v[4];
            p = p->prev->inv;
        } while (p != h[1][0]);
        if (h[2][1]->next==h[1][0]) {
            h[2][1]->next=h[1][0]->next;
        }
        if (h[0][2]->prev==h[1][0]) {
            h[0][2]->prev=h[1][0]->prev;
        }
        h[4][2]->set_neighbors(h[4][2]->next , h[4][2]->prev , h[2][4] , v[4] , e[2][4] , h[4][2]->face);
        h[2][4]->set_neighbors(h[2][4]->next , h[2][4]->prev , h[4][2] , v[2] , e[2][4] , h[2][4]->face);
        if (!boundary) {
            h[4][3]->set_neighbors(h[4][3]->next , h[4][3]->prev , h[3][4] , v[4] , e[3][4] , h[4][3]->face);
            h[3][4]->set_neighbors(h[3][4]->next , h[3][4]->prev , h[4][3] , v[3] , e[3][4] , h[3][4]->face);
            erase(h[0][3]);
            erase(h[3][1]);
        }
        else {
            f[1]->halfedge = h[1][0]->next;
        }
        erase(h[0][1]);
        erase(h[1][2]);
        erase(h[2][0]);
        erase(h[1][0]);
    } while (false);
    //修改边
    do {
        erase(e[0][1]);
        erase(e[1][2]);
        e[2][4]->halfedge = h[2][4];
        if (!boundary) {
            erase(e[1][3]);
            e[3][4]->halfedge = h[3][4];
        }
    } while (false);
    //修改点
    do {
        v[2]->halfedge = h[2][4];
        if (!boundary) {
            v[3]->halfedge = h[3][4];
        }
        v[4]->halfedge = h[4][2];
        v[4]->pos = (v[0]->pos + v[1]->pos) / 2;
        erase(v[1]);
    } while (false);
    //修改面
    do {
        erase(f[0]);
        if (!boundary) {
            erase(f[1]);
        }
    } while (false);
    validate();
    logger->trace("---end---");
    return v[4];
}

void HalfedgeMesh::loop_subdivide()
{
    optional<HalfedgeMeshFailure> check_result = validate();
    if (check_result.has_value()) {
        return;
    }
    logger->info(
        "subdivide object {} (ID: {}) with Loop Subdivision strategy", object.name, object.id
    );
    logger->info("original mesh: {} vertices, {} faces in total", vertices.size, faces.size);
    // Each vertex and edge of the original mesh can be associated with a vertex
    // in the new (subdivided) mesh.
    // Therefore, our strategy for computing the subdivided vertex locations is to
    // *first* compute the new positions using the connectivity of the original
    // (coarse) mesh. Navigating this mesh will be much easier than navigating
    // the new subdivided (fine) mesh, which has more elements to traverse.
    // We will then assign vertex positions in the new mesh based on the values
    // we computed for the original mesh.

    // Compute new positions for all the vertices in the input mesh using
    // the Loop subdivision rule and store them in Vertex::new_pos.
    //    At this point, we also want to mark each vertex as being a vertex of the
    //    original mesh. Use Vertex::is_new for this.

    // Next, compute the subdivided vertex positions associated with edges, and
    // store them in Edge::new_pos.

    // Next, we're going to split every edge in the mesh, in any order.
    // We're also going to distinguish subdivided edges that came from splitting
    // an edge in the original mesh from new edges by setting the boolean Edge::is_new.
    // Note that in this loop, we only want to iterate over edges of the original mesh.
    // Otherwise, we'll end up splitting edges that we just split (and the
    // loop will never end!)
    // I use a vector to store iterators of original because there are three kinds of
    // edges: original edges, edges split from original edges and newly added edges.
    // The newly added edges are marked with Edge::is_new property, so there is not
    // any other property to mark the edges I just split.

    // Now flip any new edge that connects an old and new vertex.

    // Finally, copy new vertex positions into the Vertex::pos.

    // Once we have successfully subdivided the mesh, set global_inconsistent
    // to true to trigger synchronization with GL::Mesh.
    modified = true;
    logger->info("subdivided mesh: {} vertices, {} faces in total", vertices.size, faces.size);
    logger->info("Loop Subdivision done");
    logger->info("");
    validate();
}

void HalfedgeMesh::simplify()
{
    optional<HalfedgeMeshFailure> check_result = validate();
    if (check_result.has_value()) {
        return;
    }
    logger->info("simplify object {} (ID: {})", object.name, object.id);
    logger->info("original mesh: {} vertices, {} faces", vertices.size, faces.size);
    unordered_map<Vertex*, Matrix4f> vertex_quadrics;
    unordered_map<Face*, Matrix4f>   face_quadrics;
    unordered_map<Edge*, EdgeRecord> edge_records;
    set<EdgeRecord>                  edge_queue;

    // Compute initial quadrics for each face by simply writing the plane equation
    // for the face in homogeneous coordinates. These quadrics should be stored
    // in face_quadrics

    // -> Compute an initial quadric for each vertex as the sum of the quadrics
    //    associated with the incident faces, storing it in vertex_quadrics

    // -> Build a priority queue of edges according to their quadric error cost,
    //    i.e., by building an Edge_Record for each edge and sticking it in the
    //    queue. You may want to use the above PQueue<Edge_Record> for this.

    // -> Until we reach the target edge budget, collapse the best edge. Remember
    //    to remove from the queue any edge that touches the collapsing edge
    //    BEFORE it gets collapsed, and add back into the queue any edge touching
    //    the collapsed vertex AFTER it's been collapsed. Also remember to assign
    //    a quadric to the collapsed vertex, and to pop the collapsed edge off the
    //    top of the queue.

    logger->info("simplified mesh: {} vertices, {} faces", vertices.size, faces.size);
    logger->info("simplification done\n");
    modified = true;
    validate();
}

void HalfedgeMesh::isotropic_remesh()
{
    optional<HalfedgeMeshFailure> check_result = validate();
    if (check_result.has_value()) {
        return;
    }
    logger->info(
        "remesh the object {} (ID: {}) with strategy Isotropic Remeshing", object.name, object.id
    );
    logger->info("original mesh: {} vertices, {} faces", vertices.size, faces.size);
    // Compute the mean edge length.

    // Repeat the four main steps for 5 or 6 iterations
    // -> Split edges much longer than the target length (being careful about
    //    how the loop is written!)
    // -> Collapse edges much shorter than the target length.  Here we need to
    //    be EXTRA careful about advancing the loop, because many edges may have
    //    been destroyed by a collapse (which ones?)
    // -> Now flip each edge if it improves vertex degree
    // -> Finally, apply some tangential smoothing to the vertex positions
    static const size_t iteration_limit = 5;
    set<Edge*>          selected_edges;
    for (size_t i = 0; i != iteration_limit; ++i) {
        // Split long edges.

        // Collapse short edges.

        // Flip edges.

        // Vertex averaging.
    }
    logger->info("remeshed mesh: {} vertices, {} faces\n", vertices.size, faces.size);
    modified = true;
    validate();
}
