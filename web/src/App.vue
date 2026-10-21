<script setup>
import { computed, nextTick, ref } from "vue"

const token = ref(sessionStorage.getItem("labassist-token") || "")
const username = ref(sessionStorage.getItem("labassist-username") || "")
const loginName = ref("")
const loginPassword = ref("")
const loginError = ref("")
const question = ref("")
const sending = ref(false)
const errorText = ref("")
const turns = ref([])
const threadEnd = ref(null)

const signedIn = computed(() => token.value !== "")

function logout() {
  token.value = ""
  username.value = ""
  sessionStorage.removeItem("labassist-token")
  sessionStorage.removeItem("labassist-username")
  turns.value = []
  question.value = ""
  errorText.value = ""
}

async function login() {
  loginError.value = ""
  const response = await fetch("/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      username: loginName.value,
      password: loginPassword.value,
    }),
  })
  const body = await response.json()
  if (!response.ok) {
    loginError.value = typeof body.detail === "string" ? body.detail : "登录失败"
    return
  }
  token.value = body.access_token
  username.value = loginName.value
  sessionStorage.setItem("labassist-token", body.access_token)
  sessionStorage.setItem("labassist-username", loginName.value)
  loginPassword.value = ""
}

function applyEvent(block, turn) {
  let eventName = "message"
  const dataLines = []
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim()
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim())
    }
  }
  if (dataLines.length === 0) {
    return
  }
  const payload = JSON.parse(dataLines.join("\n"))
  if (eventName === "citations") {
    turn.citations = payload
  }
  if (eventName === "delta") {
    turn.answer += payload
  }
}

async function readStream(response, turn) {
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  while (true) {
    const { value, done } = await reader.read()
    if (done) {
      break
    }
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split("\n\n")
    buffer = blocks.pop() ?? ""
    for (const block of blocks) {
      applyEvent(block, turn)
      await stickToBottom()
    }
  }
  if (buffer.trim()) {
    applyEvent(buffer, turn)
  }
}

async function stickToBottom() {
  await nextTick()
  threadEnd.value?.scrollIntoView({ block: "end" })
}

async function ask() {
  const content = question.value.trim()
  if (!content || sending.value) {
    return
  }
  question.value = ""
  errorText.value = ""
  const turn = { question: content, answer: "", citations: [], pending: true }
  turns.value.push(turn)
  sending.value = true
  await stickToBottom()
  try {
    const response = await fetch("/answers/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token.value}`,
      },
      body: JSON.stringify({ content }),
    })
    if (response.status === 401) {
      logout()
      loginError.value = "登录已失效"
      return
    }
    if (!response.ok) {
      const body = await response.json()
      errorText.value = typeof body.detail === "string" ? body.detail : "提问失败"
      return
    }
    await readStream(response, turn)
  } catch {
    errorText.value = "连接中断"
  } finally {
    turn.pending = false
    sending.value = false
    await stickToBottom()
  }
}
</script>

<template>
  <div class="bench">
    <header class="rail">
      <p class="mark">LabAssist</p>
      <p class="role">实验室文档</p>
      <p v-if="signedIn" class="who">{{ username }}</p>
      <button v-if="signedIn" class="text-button" type="button" @click="logout">退出</button>
    </header>

    <main class="sheet">
      <form v-if="!signedIn" class="gate" @submit.prevent="login">
        <h1>登录后提问</h1>
        <label>
          用户名
          <input v-model="loginName" autocomplete="username" required />
        </label>
        <label>
          密码
          <input v-model="loginPassword" type="password" autocomplete="current-password" required />
        </label>
        <p v-if="loginError" class="error">{{ loginError }}</p>
        <button type="submit">登录</button>
      </form>

      <template v-else>
        <section class="thread" aria-live="polite">
          <p v-if="turns.length === 0" class="empty">输入一个关于已入库 SOP 的问题。</p>
          <article v-for="(turn, index) in turns" :key="index" class="turn">
            <p class="question">{{ turn.question }}</p>
            <p class="answer">
              {{ turn.answer }}<span v-if="turn.pending" class="caret" aria-hidden="true"></span>
            </p>
            <ul v-if="turn.citations.length" class="labels">
              <li v-for="(citation, citationIndex) in turn.citations" :key="citationIndex">
                <p class="label-title">{{ citation.title }}</p>
                <p class="label-body">{{ citation.content }}</p>
              </li>
            </ul>
          </article>
          <p v-if="errorText" class="error">{{ errorText }}</p>
          <div ref="threadEnd"></div>
        </section>

        <form class="ask" @submit.prevent="ask">
          <label class="ask-label">
            问题
            <textarea
              v-model="question"
              rows="2"
              placeholder="例如：编号不一致时怎么办"
              :disabled="sending"
            ></textarea>
          </label>
          <button type="submit" :disabled="sending || question.trim() === ''">提问</button>
        </form>
      </template>
    </main>
  </div>
</template>
