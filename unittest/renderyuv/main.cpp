#include <wayland-client.h>
#include <wayland-egl.h>
#include <EGL/egl.h>
#include <GLES2/gl2.h>
#include <cstring>
#include <unistd.h>
#include <chrono>

void render_frame();
void update_yuv_textures(uint8_t *y_data, uint8_t *u_data, uint8_t *v_data);
void init_gl();

// Wayland全局对象
struct wl_display *display = nullptr;
struct wl_compositor *compositor = nullptr;
struct wl_shell *shell = nullptr;
struct wl_surface *surface = nullptr;

// EGL相关对象
EGLDisplay egl_display;
EGLContext egl_context;
EGLSurface egl_surface;
struct wl_egl_window *egl_window;

// 视频参数
const int VIDEO_WIDTH = 400;
const int VIDEO_HEIGHT = 300;
const int FPS = 25;
const int FRAME_INTERVAL_MS = 1000 / FPS;

// OpenGL纹理
GLuint y_texture, u_texture, v_texture;
GLuint program;

// 着色器源码
const char *vert_shader = 
    "attribute vec4 position;\n"
    "attribute vec2 texCoord;\n"
    "varying vec2 vTexCoord;\n"
    "void main() {\n"
    "    gl_Position = position;\n"
    "    vTexCoord = texCoord;\n"
    "}";

const char *frag_shader = 
    "precision mediump float;\n"
    "varying vec2 vTexCoord;\n"
    "uniform sampler2D ySampler;\n"
    "uniform sampler2D uSampler;\n"
    "uniform sampler2D vSampler;\n"
    "const mat4 yuv2rgb = mat4(\n"
    "    1.164383,  0.000000,  1.792741, -0.972945,\n"
    "    1.164383, -0.213249, -0.532909,  0.301483,\n"
    "    1.164383,  2.112402,  0.000000, -1.133402,\n"
    "    0.0, 0.0, 0.0, 1.0\n"
    ");\n"
    "void main() {\n"
    "    float y = texture2D(ySampler, vTexCoord).r;\n"
    "    float u = texture2D(uSampler, vTexCoord).r;\n"
    "    float v = texture2D(vSampler, vTexCoord).r;\n"
    "    gl_FragColor = vec4(y, u, v, 1.0) * yuv2rgb;\n"
    "}";

// Wayland回调监听器
static wl_registry_listener registry_listener = {
    [](void *data, wl_registry *registry, uint32_t id,
       const char *interface, uint32_t version) {
        if (strcmp(interface, "wl_compositor") == 0) {
            compositor = (wl_compositor*)wl_registry_bind(
                registry, id, &wl_compositor_interface, 3);
        } else if (strcmp(interface, "wl_shell") == 0) {
            shell = (wl_shell*)wl_registry_bind(
                registry, id, &wl_shell_interface, 1);
        }
    },
    [](void *data, wl_registry *registry, uint32_t id) {}
};

static wl_callback_listener frame_listener = {
    [](void *data, wl_callback *callback, uint32_t time) {
        static auto last_frame = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_frame).count();

        if (elapsed >= FRAME_INTERVAL_MS) {
            render_frame();
            last_frame = now;
        }

        wl_callback_destroy(callback);
        wl_callback *new_cb = wl_surface_frame(surface);
        wl_callback_add_listener(new_cb, &frame_listener, nullptr);
        wl_surface_commit(surface);
    }
};

GLuint create_shader(GLenum type, const char *source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

void init_gl() {
    // 创建着色器程序
    GLuint vert = create_shader(GL_VERTEX_SHADER, vert_shader);
    GLuint frag = create_shader(GL_FRAGMENT_SHADER, frag_shader);
    
    program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    glUseProgram(program);

    // 创建YUV纹理
    glGenTextures(1, &y_texture);
    glGenTextures(1, &u_texture);
    glGenTextures(1, &v_texture);

    auto init_tex = [](GLuint tex, int w, int h) {
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, 
                    GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
    };

    init_tex(y_texture, VIDEO_WIDTH, VIDEO_HEIGHT);
    init_tex(u_texture, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);
    init_tex(v_texture, VIDEO_WIDTH/2, VIDEO_HEIGHT/2);

    // 设置顶点数据
    GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f, 0.0f
    };
    GLfloat texCoords[] = {
        0.0f, 1.0f,
        1.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLuint vbo[2];
    glGenBuffers(2, vbo);
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    GLint posLoc = glGetAttribLocation(program, "position");
    glEnableVertexAttribArray(posLoc);
    glVertexAttribPointer(posLoc, 3, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texCoords), texCoords, GL_STATIC_DRAW);
    GLint texLoc = glGetAttribLocation(program, "texCoord");
    glEnableVertexAttribArray(texLoc);
    glVertexAttribPointer(texLoc, 2, GL_FLOAT, GL_FALSE, 0, 0);
}

void update_yuv_textures(uint8_t *y_data, uint8_t *u_data, uint8_t *v_data) {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, y_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
                   VIDEO_WIDTH, VIDEO_HEIGHT,
                   GL_LUMINANCE, GL_UNSIGNED_BYTE, y_data);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, u_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
                   VIDEO_WIDTH/2, VIDEO_HEIGHT/2,
                   GL_LUMINANCE, GL_UNSIGNED_BYTE, u_data);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, v_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 
                   VIDEO_WIDTH/2, VIDEO_HEIGHT/2,
                   GL_LUMINANCE, GL_UNSIGNED_BYTE, v_data);

    glUniform1i(glGetUniformLocation(program, "ySampler"), 0);
    glUniform1i(glGetUniformLocation(program, "uSampler"), 1);
    glUniform1i(glGetUniformLocation(program, "vSampler"), 2);
}

void render_frame() {
    // 生成测试YUV数据（实际使用时替换为真实数据）
    static uint8_t y_data[VIDEO_WIDTH * VIDEO_HEIGHT];
    static uint8_t u_data[VIDEO_WIDTH/2 * VIDEO_HEIGHT/2];
    static uint8_t v_data[VIDEO_WIDTH/2 * VIDEO_HEIGHT/2];
    
    // 填充测试图案（棋盘格）
    static int counter = 0;
    memset(y_data, counter++ % 256, sizeof(y_data));
    memset(u_data, 128, sizeof(u_data));
    memset(v_data, 128, sizeof(v_data));

    update_yuv_textures(y_data, u_data, v_data);
    
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    eglSwapBuffers(egl_display, egl_surface);
}


int main() {

    display = wl_display_connect(nullptr);
    wl_registry *registry = wl_display_get_registry(display);
    wl_registry_add_listener(registry, &registry_listener, nullptr);
    wl_display_roundtrip(display);

    // 创建窗口表面
    surface = wl_compositor_create_surface(compositor);
    wl_shell_surface *shell_surface = wl_shell_get_shell_surface(shell, surface);
    wl_shell_surface_set_toplevel(shell_surface);

    // 初始化EGL
    egl_display = eglGetDisplay((EGLNativeDisplayType)display);
    eglInitialize(egl_display, nullptr, nullptr);
    
    EGLConfig config;
    EGLint num_config;
    const EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_NONE
    };
    eglChooseConfig(egl_display, attribs, &config, 1, &num_config);
    
    eglBindAPI(EGL_OPENGL_ES_API);
    const EGLint ctx_attribs[] = { EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE };
    egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, ctx_attribs);
    
    egl_window = wl_egl_window_create(surface, VIDEO_WIDTH, VIDEO_HEIGHT);
    egl_surface = eglCreateWindowSurface(egl_display, config, egl_window, nullptr);
    eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context);

    init_gl();

    // 启动帧回调
    wl_callback *callback = wl_surface_frame(surface);
    wl_callback_add_listener(callback, &frame_listener, nullptr);
    wl_surface_commit(surface);

    while (wl_display_dispatch(display) != -1) {
        // 主事件循环
    }

    // 清理资源
    wl_egl_window_destroy(egl_window);
    eglDestroySurface(egl_display, egl_surface);
    eglDestroyContext(egl_display, egl_context);
    wl_display_disconnect(display);
    return 0;
}