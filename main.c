/*
Software Triangle Rasterizer

Resources:
https://fgiesen.wordpress.com/2013/02/17/optimizing-sw-occlusion-culling-index/

*/

#define GL_SILENCE_DEPRECATION
#include "GLFW/glfw3.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Struct definitions

typedef struct _Buffer {
    int32_t *data;
    uint32_t width;
    uint32_t height;
} Buffer;

typedef struct _Vec2i {
    int32_t x;
    int32_t y;
} Vec2i;

typedef struct _Vec4 {
    float x;
    float y;
    float z;
    float w;
} Vec4;

typedef struct _Mat4x4 {
    float x[16];
} Mat4x4;

typedef struct _Vertex {
    Vec4 p;
    float u;
    float v;
} Vertex;

// Helper math functions

Mat4x4 get_perspective_projection_matrix(float aspect_ratio, float fov, float near, float far) {
    Mat4x4 result = {0};

    // X and Y coordinates projected to [-1, 1] range (after perspective divide)
    // Z coordinate projected to [0, 1] range (after perspective divide)
    float tanf = tan(fov / 2.0f);
    result.x[0] = 1.0f / (aspect_ratio * tanf);
    result.x[5] = 1.0f / tanf;
    result.x[10] = -far / (far - near);
    result.x[11] = -1;
    result.x[14] = near * far / (far - near);

    return result;
}

Mat4x4 get_rotation_matrix_y(float angle) {
    Mat4x4 result = {0};

    result.x[0] = cos(angle);
    result.x[2] = -sin(angle);
    result.x[5] = 1;
    result.x[8] = sin(angle);
    result.x[10] = cos(angle);
    result.x[15] = 1;

    return result;
}

Mat4x4 get_translation_matrix(float x, float y, float z) {
    Mat4x4 result = {0};

    result.x[0] = 1;
    result.x[5] = 1;
    result.x[10] = 1;
    result.x[12] = x;
    result.x[13] = y;
    result.x[14] = z;
    result.x[15] = 1;

    return result;
}

Mat4x4 matmul(Mat4x4 a, Mat4x4 b) {
    Mat4x4 result = {0};

    result.x[0] = a.x[0] * b.x[0] + a.x[4] * b.x[1] + a.x[8] * b.x[2] + a.x[12] * b.x[3];
    result.x[1] = a.x[1] * b.x[0] + a.x[5] * b.x[1] + a.x[9] * b.x[2] + a.x[13] * b.x[3];
    result.x[2] = a.x[2] * b.x[0] + a.x[6] * b.x[1] + a.x[10] * b.x[2] + a.x[14] * b.x[3];
    result.x[3] = a.x[3] * b.x[0] + a.x[7] * b.x[1] + a.x[11] * b.x[2] + a.x[15] * b.x[3];

    result.x[4] = a.x[0] * b.x[4] + a.x[4] * b.x[5] + a.x[8] * b.x[6] + a.x[12] * b.x[7];
    result.x[5] = a.x[1] * b.x[4] + a.x[5] * b.x[5] + a.x[9] * b.x[6] + a.x[13] * b.x[7];
    result.x[6] = a.x[2] * b.x[4] + a.x[6] * b.x[5] + a.x[10] * b.x[6] + a.x[14] * b.x[7];
    result.x[7] = a.x[3] * b.x[4] + a.x[7] * b.x[5] + a.x[11] * b.x[6] + a.x[15] * b.x[7];

    result.x[8] = a.x[0] * b.x[8] + a.x[4] * b.x[9] + a.x[8] * b.x[10] + a.x[12] * b.x[11];
    result.x[9] = a.x[1] * b.x[8] + a.x[5] * b.x[9] + a.x[9] * b.x[10] + a.x[13] * b.x[11];
    result.x[10] = a.x[2] * b.x[8] + a.x[6] * b.x[9] + a.x[10] * b.x[10] + a.x[14] * b.x[11];
    result.x[11] = a.x[3] * b.x[8] + a.x[7] * b.x[9] + a.x[11] * b.x[10] + a.x[15] * b.x[11];

    result.x[12] = a.x[0] * b.x[12] + a.x[4] * b.x[13] + a.x[8] * b.x[14] + a.x[12] * b.x[15];
    result.x[13] = a.x[1] * b.x[12] + a.x[5] * b.x[13] + a.x[9] * b.x[14] + a.x[13] * b.x[15];
    result.x[14] = a.x[2] * b.x[12] + a.x[6] * b.x[13] + a.x[10] * b.x[14] + a.x[14] * b.x[15];
    result.x[15] = a.x[3] * b.x[12] + a.x[7] * b.x[13] + a.x[11] * b.x[14] + a.x[15] * b.x[15];

    return result;
}

Vec4 vecmatmul(Mat4x4 m, Vec4 v) {
    Vec4 result;

    result.x = v.x * m.x[0] + v.y * m.x[4] + v.z * m.x[8] + v.w * m.x[12];
    result.y = v.x * m.x[1] + v.y * m.x[5] + v.z * m.x[9] + v.w * m.x[13];
    result.z = v.x * m.x[2] + v.y * m.x[6] + v.z * m.x[10] + v.w * m.x[14];
    result.w = v.x * m.x[3] + v.y * m.x[7] + v.z * m.x[11] + v.w * m.x[15];

    return result;
}

int32_t min(int32_t a, int32_t b) {
    return a <= b ? a : b;
}

int32_t max(int32_t a, int32_t b) {
    return a >= b ? a : b;
}

// Rendering functions

void set_pixel(uint32_t x, uint32_t y, uint8_t value, Buffer *buffer) {
    int32_t pixel_value = ((int32_t)255 << 24) | ((int32_t)value << 16) | ((int32_t)value << 8) | (int32_t)value;
    buffer->data[x + y * buffer->width] = pixel_value;
}

// Uses this fill rule:
// https://docs.microsoft.com/en-us/windows/win32/direct3d11/d3d10-graphics-programming-guide-rasterizer-stage-rules
bool is_top_left(Vec2i v1, Vec2i v2) {
    // Check if edge is a top edge.
    // Top edge is an edge that is completely horizontal and is above other edges.
    // Since we're assuming counter clockwise vertex ordering top, the edge that is
    // at the top has to go to the left.
    if (v1.y == v2.y && v2.x < v1.x) {
        return true;
    }

    // Check if edge is a left edge.
    // Left edge is an edge that is on the left side of the triangle.
    // Again, since we assume counter clockwise vertex order, the left edge has
    // to point downwards.
    if (v2.y < v1.y) {
        return true;
    }

    // The edge is neither top or left.
    return false;
}

void rasterize_triangle(Vertex vertices[3], Buffer *buffer) {
    // Get pixel coordiantes for each vertex
    Vec2i p1 = {
        (int)(vertices[0].p.x * buffer->width / 2 + buffer->width / 2),
        (int)(vertices[0].p.y * buffer->height / 2 + buffer->height / 2)
    };
    Vec2i p2 = {
        (int)(vertices[1].p.x * buffer->width / 2 + buffer->width / 2),
        (int)(vertices[1].p.y * buffer->height / 2 + buffer->height / 2)
    };
    Vec2i p3 = {
        (int)(vertices[2].p.x * buffer->width / 2 + buffer->width / 2),
        (int)(vertices[2].p.y * buffer->height / 2 + buffer->height / 2)
    };

    // Get depth for each vertex
    float z1 = vertices[0].p.w;
    float z2 = vertices[1].p.w;
    float z3 = vertices[2].p.w;

    // Get U texture coordinate for each vertex.
    float u1 = vertices[0].u;
    float u2 = vertices[1].u;
    float u3 = vertices[2].u;

    // Get V texture coordinate for each vertex.
    float v1 = vertices[0].v;
    float v2 = vertices[1].v;
    float v3 = vertices[2].v;

    // Compute triangle bbox
    int32_t min_x = min(p1.x, min(p2.x, p3.x));
    int32_t max_x = max(p1.x, max(p2.x, p3.x));
    int32_t min_y = min(p1.y, min(p2.y, p3.y));
    int32_t max_y = max(p1.y, max(p2.y, p3.y));

    // Clipping
    min_x = max(min_x, 0);
    min_y = max(min_y, 0);
    max_x = min(max_x, buffer->width - 1);
    max_y = min(max_y, buffer->height - 1);

    // Pre-compute twice the triangle area. This is used to normalize barycentric coordinates.
    // See comment next to edge functions below for more details.
    int32_t double_triangle_area = (p3.y - p1.y) * (p2.x - p1.x) + (p3.x - p1.x) * (p2.y - p1.y); 

    // Rasterization loop
    for(int32_t y = min_y; y < max_y; ++y) {
        for(int32_t x = min_x; x < max_x; ++x) {
            // Compute edge functions.
            // NOTE: We're assuming counter clockwise vertex ordering, so we're testing
            // if the vertex is to the left of the edge. One way to do this is to
            // rotate the edge by 90 degrees to the left (x, y) -> (-y, x) and then
            // compute dot product of point relative to the edge's first vertex with
            // the rotated edge. Point is to the left if dot product is > 0.
            // NOTE: Edge 1 (e1) is edge between vertices 2 and 3,
            // e2 between v3 and v1 and e3 between v1 and v2.
            // NOTE: These are also unnormalized barycentric coordinates (lucky us).
            // To get normalized coordinates, we'd have to divide by 2x triangle's area.
            // To see why, consider edge functions evaluated at the vertex points.
            //
            //   v2r 
            //    |    v3
            //    |   /|
            // e3r|  /a|  
            //    |a/  |
            //    |/   |
            //   v1 ----------- v2
            //          e3
            //
            // Edge function for edge 3 (e3 = v1-v2) is computed as dot product between rotated
            // edge 3 - e3r and edge 2 (e2 = v3-v1). The dot product is |e3r| * |e2| * cos(a).
            // From the picture, we can see that |e2| * cos(a) is actually the height of
            // the triangle, which leaves the dot product at |e3r| * h = |e3| * h, which is
            // twice the area of the triangle. 
            int32_t e1 = (y - p2.y) * (p3.x - p2.x) - (x - p2.x) * (p3.y - p2.y);
            int32_t e2 = (y - p3.y) * (p1.x - p3.x) - (x - p3.x) * (p1.y - p3.y);
            int32_t e3 = (y - p1.y) * (p2.x - p1.x) - (x - p1.x) * (p2.y - p1.y);

            // Compute barycentric coordinates. See comment above for explanation.
            float b1 = (float)e1 / (float)double_triangle_area;
            float b2 = (float)e2 / (float)double_triangle_area;
            float b3 = (float)e3 / (float)double_triangle_area;

            // To implement our fill rule we need to decide if the edge is either top or left
            // (check the is_top_left function for details). In case it is, we'll bias the
            // edge function so we either include or exclude pixels on the edge.
            e1 += is_top_left(p2, p3) ? 0 : -1; 
            e2 += is_top_left(p3, p1) ? 0 : -1; 
            e3 += is_top_left(p1, p2) ? 0 : -1;

            // Interpolate depth using barycentric coordinates.
            float z = 1.0f / (b1 / z1 + b2 / z2 + b3 / z3);

            // Interpolate uv texture coordinates using barycentric coordinates and depth.
            float u = z * (b1 * u1 / z1 + b2 * u2 / z2 + b3 * u3 / z3);
            float v = z * (b1 * v1 / z1 + b2 * v2 / z2 + b3 * v3 / z3);

            // Check if all the edge functions have positive sign. We can just OR them together
            // and check if they're still positive number - if even one of them is negative,
            // the sign bit will be set to 1 so the result after bitwise OR will be negative as well.
            if((e1 | e2 | e3) >= 0) {
                // "Grid Texture" based on UV coordinates
                const float grid_size = 0.05f;
                bool is_u_grid = fmod(u + grid_size / 8.0f, grid_size) < grid_size / 4.0f;
                bool is_v_grid = fmod(v + grid_size / 8.0f, grid_size) < grid_size / 4.0f;
                bool is_grid = is_u_grid || is_v_grid;

                uint8_t color = is_grid ? 255 : 64;
                set_pixel(x, y, color, buffer);
            }
        }
    }
}

Vertex project_vertex(Vertex v, Mat4x4 transform_matrix) {
    // Apply transformation matrix
    Vec4 projected_position = vecmatmul(transform_matrix, v.p);

    // Perspective divide
    projected_position.x /= projected_position.w;
    projected_position.y /= projected_position.w;  
    projected_position.z /= projected_position.w;  

    Vertex result = {};
    result.u = v.u;
    result.v = v.v;
    result.p = projected_position;
    return result;
}

// Main program

int main(int argc, char **argv) {
    // Initialize GLFW.
    int glfw_init_status = glfwInit();
    if (glfw_init_status == GLFW_FALSE) {
        printf("GLFW initialization not successful!\n");
    }

    // Get a GLFW window.
    uint32_t window_width = 800;
    uint32_t window_height = 800;
    GLFWwindow *window = glfwCreateWindow(window_width, window_height, "Rasterizer", NULL, NULL);
    if (window == NULL) {
        glfwTerminate();
    }
    glfwMakeContextCurrent(window);

    // Compute GL's framebuffer size.
    // The size is different from window size due to window content scaling.
    float x_scale = 0.0;
    float y_scale = 0.0;
    glfwGetWindowContentScale(window, &x_scale, &y_scale);
    uint32_t gl_buffer_width = window_width * x_scale;
    uint32_t gl_buffer_height = window_height * y_scale;

    // Initialize frame buffer.
    Buffer buffer;
    buffer.width = 800;
    buffer.height = 800;
    buffer.data = (int32_t *)malloc(sizeof(int32_t) * buffer.width * buffer.height);
    if(!buffer.data) {
        printf("Couldn't allocate memory for frame buffer for some reason.");
    }
    
    // Compute scaling needed to display our buffer in the GL buffer.
    float buffer_scale_x = (float)gl_buffer_width / (float)buffer.width;
    float buffer_scale_y = (float)gl_buffer_height / (float)buffer.height;

    // Set up a projection matrix.
    Mat4x4 projection_matrix = get_perspective_projection_matrix(1.0f, 1.04f, -1, -10.0f);

    // Render loop.
    while (!glfwWindowShouldClose(window)) {
        // Event handling.
        glfwPollEvents();

        // Check for ESC key press.
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            break;
        }

        // Quad vertices.
        Vertex v1 = {{-0.5f, -0.5f, 0, 1}, 0.0f, 0.0f};
        Vertex v2 = {{0.5f, -0.5f, 0, 1}, 1.0f, 0.0f};
        Vertex v3 = {{0.5f, 0.5f, 0, 1}, 1.0f, 1.0f};
        Vertex v4 = {{-0.5f, 0.5f, 0, 1}, 0.0f, 1.0f};
        
        // Set up current transformation matrix.
        // We're just rotating the camera around Y-axis and moving it slightly further from camera. 
        double current_time = glfwGetTime();
        Mat4x4 rotation_matrix = get_rotation_matrix_y(current_time * 0.5f);
        Mat4x4 translation_matrix = get_translation_matrix(0, 0, -2);
        Mat4x4 transform_matrix = matmul(projection_matrix, matmul(translation_matrix, rotation_matrix));
        
        // Project vertices into NDC.
        v1 = project_vertex(v1, transform_matrix);
        v2 = project_vertex(v2, transform_matrix);
        v3 = project_vertex(v3, transform_matrix);
        v4 = project_vertex(v4, transform_matrix);

        // Clear screen.
        memset(buffer.data, 0, sizeof(uint32_t) * buffer.width * buffer.height);

        // Render two quads, so we see both front and back faces.
        rasterize_triangle((Vertex[3]){v1, v2, v3}, &buffer);  // Front face
        rasterize_triangle((Vertex[3]){v1, v3, v4}, &buffer);        
        rasterize_triangle((Vertex[3]){v3, v2, v1}, &buffer);  // Back face
        rasterize_triangle((Vertex[3]){v4, v3, v1}, &buffer);        

        // Show the frame buffer.
        glClear(GL_COLOR_BUFFER_BIT);
        glPixelZoom(buffer_scale_x, buffer_scale_y);
        glDrawPixels(
            buffer.width, buffer.height,
            GL_RGBA, GL_UNSIGNED_BYTE, &buffer.data[0]
        );
        glfwSwapBuffers(window);
    }

    // Exit
    glfwTerminate();
}