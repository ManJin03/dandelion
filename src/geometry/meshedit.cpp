#include "halfedge.h"

#include <set>
#include <vector>
#include <string>

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
    logger->trace("---start flip_edge {}---" , e->id);

    if (e->on_boundary()) {
        logger->trace("boundary edge");
        return e;
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
    //所需边
    auto h12 = e->halfedge; //h43
    auto h21 = h12->inv; //h34
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
    f1->halfedge = h12;
    f2->halfedge = h21;
    //翻转所选边
    h12->set_neighbors(h31 , h14 , h21 , v4 , e , f1);
    h21->set_neighbors(h42 , h23 , h12 , v3 , e , f2);
    //修正其余边
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
    //修正点的边
    v1->halfedge = h14;
    v2->halfedge = h23;
    v3->halfedge = h31;
    v4->halfedge = h42;
    //结束日志
    printface(f1);
    printface(f2);
    logger->trace("---end---");
    return e;
}

optional<Vertex*> HalfedgeMesh::split_edge(Edge* e)
{
    (void)e;
    return std::nullopt;
}

optional<Vertex*> HalfedgeMesh::collapse_edge(Edge* e)
{
    (void)e;
    return std::nullopt;
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
