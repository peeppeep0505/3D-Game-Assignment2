

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <string>
#include <cmath>

// ─────────────────────────────────────────────────────────────────────────────
//  Inline shader sources
// ─────────────────────────────────────────────────────────────────────────────

static const char* VERT_SRC = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos     = vec3(model * vec4(aPos, 1.0));
    Normal      = mat3(transpose(inverse(model))) * aNormal;
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
)GLSL";

static const char* FRAG_SRC = R"GLSL(
#version 330 core
out vec4 FragColor;

in vec3 FragPos;
in vec3 Normal;

#define NR_POINT_LIGHTS 4

struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {
    vec3  position;
    float constant;
    float linear;
    float quadratic;
    vec3  ambient;
    vec3  diffuse;
    vec3  specular;
};

struct SpotLight {
    vec3  position;
    vec3  direction;
    float cutOff;
    float outerCutOff;
    float constant;
    float linear;
    float quadratic;
    vec3  ambient;
    vec3  diffuse;
    vec3  specular;
};

uniform vec3       viewPos;
uniform DirLight   dirLight;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform SpotLight  spotLight;
uniform vec3       matDiffuse;
uniform vec3       matSpecular;
uniform float      matShininess;

vec3 CalcDirLight(DirLight L, vec3 n, vec3 v)
{
    vec3  d    = normalize(-L.direction);
    float diff = max(dot(n, d), 0.0);
    vec3  r    = reflect(-d, n);
    float spec = pow(max(dot(v, r), 0.0), matShininess);
    return L.ambient * matDiffuse
         + L.diffuse  * diff * matDiffuse
         + L.specular * spec * matSpecular;
}

vec3 CalcPointLight(PointLight L, vec3 n, vec3 fp, vec3 v)
{
    vec3  d    = normalize(L.position - fp);
    float diff = max(dot(n, d), 0.0);
    vec3  r    = reflect(-d, n);
    float spec = pow(max(dot(v, r), 0.0), matShininess);
    float dist = length(L.position - fp);
    float att  = 1.0 / (L.constant + L.linear*dist + L.quadratic*dist*dist);
    return (L.ambient * matDiffuse
          + L.diffuse  * diff * matDiffuse
          + L.specular * spec * matSpecular) * att;
}

vec3 CalcSpotLight(SpotLight L, vec3 n, vec3 fp, vec3 v)
{
    vec3  d        = normalize(L.position - fp);
    float diff     = max(dot(n, d), 0.0);
    vec3  r        = reflect(-d, n);
    float spec     = pow(max(dot(v, r), 0.0), matShininess);
    float dist     = length(L.position - fp);
    float att      = 1.0 / (L.constant + L.linear*dist + L.quadratic*dist*dist);
    float theta    = dot(d, normalize(-L.direction));
    float eps      = L.cutOff - L.outerCutOff;
    float inten    = clamp((theta - L.outerCutOff) / eps, 0.0, 1.0);
    return (L.ambient * matDiffuse
          + L.diffuse  * diff * matDiffuse
          + L.specular * spec * matSpecular) * att * inten;
}

void main()
{
    vec3 n = normalize(Normal);
    vec3 v = normalize(viewPos - FragPos);

    vec3 c = CalcDirLight(dirLight, n, v);
    for (int i = 0; i < NR_POINT_LIGHTS; i++)
        c += CalcPointLight(pointLights[i], n, FragPos, v);
    c += CalcSpotLight(spotLight, n, FragPos, v);

    FragColor = vec4(c, 1.0);
}
)GLSL";

// Simple flat-colour shader for the light-marker cubes
static const char* LIGHT_VERT = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)GLSL";

static const char* LIGHT_FRAG = R"GLSL(
#version 330 core
out vec4 FragColor;
uniform vec3 lightColor;
void main()
{
    FragColor = vec4(lightColor, 1.0);
}
)GLSL";

// ─────────────────────────────────────────────────────────────────────────────
//  Shader helper
// ─────────────────────────────────────────────────────────────────────────────
static GLuint compileShader(GLenum type, const char* src)
{
    GLuint id = glCreateShader(type);
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);
    GLint ok; glGetShaderiv(id, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetShaderInfoLog(id, 1024, nullptr, log);
        std::cerr << "[SHADER ERROR] " << log << "\n";
    }
    return id;
}

static GLuint makeProgram(const char* vsrc, const char* fsrc)
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, vsrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs);
    glLinkProgram(prog);
    GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024]; glGetProgramInfoLog(prog, 1024, nullptr, log);
        std::cerr << "[LINK ERROR] " << log << "\n";
    }
    glDeleteShader(vs); glDeleteShader(fs);
    return prog;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Uniform setters
// ─────────────────────────────────────────────────────────────────────────────
static void setFloat(GLuint p, const char* n, float v) { glUniform1f(glGetUniformLocation(p, n), v); }
static void setVec3(GLuint p, const char* n, glm::vec3 v) { glUniform3fv(glGetUniformLocation(p, n), 1, glm::value_ptr(v)); }
static void setMat4(GLuint p, const char* n, const glm::mat4& m) { glUniformMatrix4fv(glGetUniformLocation(p, n), 1, GL_FALSE, glm::value_ptr(m)); }

// ─────────────────────────────────────────────────────────────────────────────
//  Camera (simple FPS)
// ─────────────────────────────────────────────────────────────────────────────
struct Camera {
    glm::vec3 pos = { 0,8,20 };
    glm::vec3 front = { 0,0,-1 };
    glm::vec3 up = { 0,1,0 };
    float yaw = -90, pitch = 0, zoom = 45;

    glm::mat4 view() const { return glm::lookAt(pos, pos + front, up); }

    void moveForward(float dt) { pos += front * (5.0f * dt); }
    void moveBackward(float dt) { pos -= front * (5.0f * dt); }
    void moveLeft(float dt) { pos -= glm::normalize(glm::cross(front, up)) * (5.0f * dt); }
    void moveRight(float dt) { pos += glm::normalize(glm::cross(front, up)) * (5.0f * dt); }

    void look(float dx, float dy)
    {
        yaw += dx * 0.1f;
        pitch = glm::clamp(pitch + dy * 0.1f, -89.0f, 89.0f);
        front = glm::normalize(glm::vec3(
            cosf(glm::radians(yaw)) * cosf(glm::radians(pitch)),
            sinf(glm::radians(pitch)),
            sinf(glm::radians(yaw)) * cosf(glm::radians(pitch))));
    }
} cam;

// ─────────────────────────────────────────────────────────────────────────────
//  Globals
// ─────────────────────────────────────────────────────────────────────────────
const int SCR_W = 1280, SCR_H = 720;
float lastX = SCR_W / 2.f, lastY = SCR_H / 2.f;
bool  firstMouse = true;
float dt = 0, lastFrame = 0;
bool  paused = false;
float animTime = 0;

void framebuffer_size_callback(GLFWwindow*, int w, int h) { glViewport(0, 0, w, h); }

void mouse_callback(GLFWwindow*, double xd, double yd)
{
    float x = (float)xd, y = (float)yd;
    if (firstMouse) { lastX = x; lastY = y; firstMouse = false; }
    cam.look(x - lastX, lastY - y);
    lastX = x; lastY = y;
}

void scroll_callback(GLFWwindow*, double, double yo)
{
    cam.zoom = glm::clamp(cam.zoom - (float)yo, 1.0f, 90.0f);
}

void processInput(GLFWwindow* w)
{
    if (glfwGetKey(w, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(w, true);
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) cam.moveForward(dt);
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) cam.moveBackward(dt);
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) cam.moveLeft(dt);
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) cam.moveRight(dt);

    static int prev = GLFW_RELEASE;
    int cur = glfwGetKey(w, GLFW_KEY_SPACE);
    if (cur == GLFW_PRESS && prev == GLFW_RELEASE) paused = !paused;
    prev = cur;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* win = glfwCreateWindow(SCR_W, SCR_H, "Kinetic Sculpture", nullptr, nullptr);
    if (!win) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glfwSetFramebufferSizeCallback(win, framebuffer_size_callback);
    glfwSetCursorPosCallback(win, mouse_callback);
    glfwSetScrollCallback(win, scroll_callback);
    glfwSetInputMode(win, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "GLAD failed\n"; return -1;
    }

    // Print GL version so we can confirm context
    std::cout << "OpenGL: " << glGetString(GL_VERSION) << "\n";

    glEnable(GL_DEPTH_TEST);

    // Build programs
    GLuint prog = makeProgram(VERT_SRC, FRAG_SRC);
    GLuint lightProg = makeProgram(LIGHT_VERT, LIGHT_FRAG);

    // ── Cube: pos(3) + normal(3), stride = 6 floats ───────────────────────────
    float verts[] = {
        // Back
        -0.5f,-0.5f,-0.5f,  0, 0,-1,
         0.5f, 0.5f,-0.5f,  0, 0,-1,
         0.5f,-0.5f,-0.5f,  0, 0,-1,
         0.5f, 0.5f,-0.5f,  0, 0,-1,
        -0.5f,-0.5f,-0.5f,  0, 0,-1,
        -0.5f, 0.5f,-0.5f,  0, 0,-1,
        // Front
        -0.5f,-0.5f, 0.5f,  0, 0, 1,
         0.5f,-0.5f, 0.5f,  0, 0, 1,
         0.5f, 0.5f, 0.5f,  0, 0, 1,
         0.5f, 0.5f, 0.5f,  0, 0, 1,
        -0.5f, 0.5f, 0.5f,  0, 0, 1,
        -0.5f,-0.5f, 0.5f,  0, 0, 1,
        // Left
        -0.5f, 0.5f, 0.5f, -1, 0, 0,
        -0.5f, 0.5f,-0.5f, -1, 0, 0,
        -0.5f,-0.5f,-0.5f, -1, 0, 0,
        -0.5f,-0.5f,-0.5f, -1, 0, 0,
        -0.5f,-0.5f, 0.5f, -1, 0, 0,
        -0.5f, 0.5f, 0.5f, -1, 0, 0,
        // Right
         0.5f, 0.5f, 0.5f,  1, 0, 0,
         0.5f,-0.5f,-0.5f,  1, 0, 0,
         0.5f, 0.5f,-0.5f,  1, 0, 0,
         0.5f,-0.5f,-0.5f,  1, 0, 0,
         0.5f, 0.5f, 0.5f,  1, 0, 0,
         0.5f,-0.5f, 0.5f,  1, 0, 0,
         // Bottom
         -0.5f,-0.5f,-0.5f,  0,-1, 0,
          0.5f,-0.5f,-0.5f,  0,-1, 0,
          0.5f,-0.5f, 0.5f,  0,-1, 0,
          0.5f,-0.5f, 0.5f,  0,-1, 0,
         -0.5f,-0.5f, 0.5f,  0,-1, 0,
         -0.5f,-0.5f,-0.5f,  0,-1, 0,
         // Top
         -0.5f, 0.5f,-0.5f,  0, 1, 0,
          0.5f, 0.5f, 0.5f,  0, 1, 0,
          0.5f, 0.5f,-0.5f,  0, 1, 0,
          0.5f, 0.5f, 0.5f,  0, 1, 0,
         -0.5f, 0.5f,-0.5f,  0, 1, 0,
         -0.5f, 0.5f, 0.5f,  0, 1, 0,
    };

    GLuint VBO, cubeVAO, lightVAO;
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    // Sculpture cubes
    glGenVertexArrays(1, &cubeVAO);
    glBindVertexArray(cubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Light markers
    glGenVertexArrays(1, &lightVAO);
    glBindVertexArray(lightVAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // ── Point light config ────────────────────────────────────────────────────
    const float OR[4] = { 8,11, 9, 6.5f };
    const float OY[4] = { 3, 1.5f, 5, 2.5f };
    const float SP[4] = { 0.7f,-0.5f,1.1f,-0.9f };
    glm::vec3 PC[4] = { {1,.25f,.25f},{.25f,1,.25f},{.25f,.25f,1},{1,.8f,.2f} };

    const int   GRID = 10;
    const float SPACING = 2.2f;

    // ── Render loop ───────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(win))
    {
        float now = (float)glfwGetTime();
        dt = now - lastFrame; lastFrame = now;
        if (!paused) animTime += dt;

        processInput(win);

        glClearColor(0.04f, 0.04f, 0.08f, 1);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glm::mat4 proj = glm::perspective(glm::radians(cam.zoom),
            (float)SCR_W / SCR_H, 0.1f, 120.f);
        glm::mat4 view = cam.view();

        // Point light positions
        glm::vec3 ptPos[4];
        for (int i = 0; i < 4; i++) {
            float a = SP[i] * animTime + i * glm::two_pi<float>() / 4.f;
            ptPos[i] = { OR[i] * cosf(a),
                        OY[i] + 1.5f * sinf(animTime * .7f + i),
                        OR[i] * sinf(a) };
        }

        // ── Lighting pass ──────────────────────────────────────────────────
        glUseProgram(prog);
        setMat4(prog, "projection", proj);
        setMat4(prog, "view", view);
        setVec3(prog, "viewPos", cam.pos);

        // Material
        setVec3(prog, "matDiffuse", { 0.2f,0.45f,0.7f });
        setVec3(prog, "matSpecular", { 0.8f,0.85f,0.9f });
        setFloat(prog, "matShininess", 96.f);

        // Directional
        setVec3(prog, "dirLight.direction", { -0.3f,-1,-0.4f });
        setVec3(prog, "dirLight.ambient", { 0.04f,0.04f,0.06f });
        setVec3(prog, "dirLight.diffuse", { 0.2f,0.2f,0.3f });
        setVec3(prog, "dirLight.specular", { 0.5f,0.5f,0.5f });

        // Point lights
        for (int i = 0; i < 4; i++) {
            std::string b = "pointLights[" + std::to_string(i) + "].";
            setVec3(prog, (b + "position").c_str(), ptPos[i]);
            setFloat(prog, (b + "constant").c_str(), 1.f);
            setFloat(prog, (b + "linear").c_str(), 0.07f);
            setFloat(prog, (b + "quadratic").c_str(), 0.017f);
            setVec3(prog, (b + "ambient").c_str(), PC[i] * 0.05f);
            setVec3(prog, (b + "diffuse").c_str(), PC[i]);
            setVec3(prog, (b + "specular").c_str(), PC[i]);
        }

        // Spot
        setVec3(prog, "spotLight.position", cam.pos);
        setVec3(prog, "spotLight.direction", cam.front);
        setFloat(prog, "spotLight.cutOff", cosf(glm::radians(12.5f)));
        setFloat(prog, "spotLight.outerCutOff", cosf(glm::radians(17.5f)));
        setFloat(prog, "spotLight.constant", 1.f);
        setFloat(prog, "spotLight.linear", 0.05f);
        setFloat(prog, "spotLight.quadratic", 0.012f);
        setVec3(prog, "spotLight.ambient", { 0,0,0 });
        setVec3(prog, "spotLight.diffuse", { 1,1,1 });
        setVec3(prog, "spotLight.specular", { 1,1,1 });

        // ── Draw sculpture ─────────────────────────────────────────────────
        glBindVertexArray(cubeVAO);
        float off = (GRID - 1) * SPACING * 0.5f;

        for (int row = 0; row < GRID; row++)
            for (int col = 0; col < GRID; col++)
            {
                float gx = col * SPACING - off;
                float gz = row * SPACING - off;
                float d = sqrtf(gx * gx + gz * gz);

                float gy = 2.0f * sinf(d * 0.55f - animTime * 2.f)
                    + 0.8f * sinf(gx * 0.5f + animTime * 1.3f)
                    + 0.8f * cosf(gz * 0.5f - animTime * 1.1f);

                float spin = animTime * 50.f + d * 12.f;
                float s = 0.88f + 0.12f * sinf(animTime * 3.f + d);

                glm::mat4 model(1.f);
                model = glm::translate(model, { gx,gy,gz });
                model = glm::rotate(model, glm::radians(spin), { 0,1,0 });
                model = glm::scale(model, { s,s,s });
                setMat4(prog, "model", model);
                glDrawArrays(GL_TRIANGLES, 0, 36);
            }

        // ── Draw light markers ─────────────────────────────────────────────
        glUseProgram(lightProg);
        setMat4(lightProg, "projection", proj);
        setMat4(lightProg, "view", view);
        glBindVertexArray(lightVAO);
        for (int i = 0; i < 4; i++) {
            setVec3(lightProg, "lightColor", PC[i]);
            glm::mat4 m = glm::scale(glm::translate(glm::mat4(1), ptPos[i]), { .25f,.25f,.25f });
            setMat4(lightProg, "model", m);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &cubeVAO);
    glDeleteVertexArrays(1, &lightVAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate();
    return 0;
}